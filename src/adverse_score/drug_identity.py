"""Drug identity resolution — Phase 1 of the PSUR consolidation rebuild.

Pure, non-agent-facing module: given a raw user-provided drug name, resolves
the canonical set of brand/generic/substance name variants to query and the
drug's market authorization (FDA approval) date, which anchors the PSUR
reporting period in Phase 2.

This module does NOT use the agent-facing payload shape (clinical_disclaimer,
diagnosis_lock, requires_human_review, system_directive) — that translation
belongs to Phase 8's agent_tools.py. Keeping this module's contract as plain
dataclasses preserves the single-responsibility split already established by
fda_client.py / prr.py / label_classifier.py.
"""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional

import requests

from .config import (
    API_TIMEOUT_DEFAULT,
    DRUGSFDA_ENDPOINT,
    NDC_ENDPOINT,
    DRUG_IDENTITY_LABEL_LIMIT,
    DRUG_IDENTITY_NDC_LIMIT,
    DRUG_IDENTITY_DRUGSFDA_LIMIT,
    DRUGSFDA_APPROVED_STATUS,
)
from .fda_client import FDAClient
from .logger import get_logger, log_event

logger = get_logger("drug_identity")

LABEL_ENDPOINT = "https://api.fda.gov/drug/label.json"


@dataclass(frozen=True)
class DrugIdentity:
    canonical_name: str
    brand_names: list
    generic_names: list
    substance_names: list
    application_numbers: list
    market_authorization_date: Optional[date]
    market_authorization_date_source: str
    resolution_confidence: Literal["EXACT", "FUZZY", "PARTIAL"]


@dataclass(frozen=True)
class DrugIdentityError:
    query: str
    reason: Literal["NOT_FOUND", "UPSTREAM_ERROR"]
    message: str
    attempted_variants: list


def resolve_drug_identity(raw_name: str, client: Optional[FDAClient] = None):
    """Resolve a raw drug name into a DrugIdentity, or a DrugIdentityError.

    Never returns an empty DrugIdentity or an empty result silently — a
    failed lookup always produces a structured DrugIdentityError.
    """
    if client is None:
        client = FDAClient()

    query = (raw_name or "").strip()
    if not query:
        return DrugIdentityError(
            query=raw_name or "",
            reason="NOT_FOUND",
            message="Drug name must not be blank.",
            attempted_variants=[],
        )

    try:
        records, confidence, attempted_variants = _resolve_name_variants(query, client)
    except requests.exceptions.RequestException as e:
        log_event(logger, "drug_identity_upstream_error", query=query, error=str(e))
        return DrugIdentityError(
            query=query, reason="UPSTREAM_ERROR",
            message=f"openFDA lookup failed for '{query}': {e}",
            attempted_variants=[],
        )

    if not records:
        log_event(logger, "drug_identity_not_found", query=query, attempted=attempted_variants)
        return DrugIdentityError(
            query=query,
            reason="NOT_FOUND",
            message=f"No FDA-registered drug found matching '{query}'. Verify spelling or try the exact brand/generic name.",
            attempted_variants=attempted_variants,
        )

    brand_names, generic_names, substance_names, application_numbers = _union_openfda_fields(records)
    canonical_name = (brand_names or generic_names or [query.upper()])[0]

    try:
        auth_date, auth_source = _lookup_market_authorization_date(
            application_numbers, brand_names, generic_names, client)
    except requests.exceptions.RequestException as e:
        # Identity itself resolved fine — only the date lookup failed upstream.
        # This is a partial success, not a DrugIdentityError.
        log_event(logger, "drug_identity_auth_date_upstream_error", query=query, error=str(e))
        auth_date, auth_source = None, "unavailable: upstream error during drugsfda.json lookup"

    resolution_confidence = confidence
    if auth_date is None and resolution_confidence == "EXACT":
        resolution_confidence = "PARTIAL"

    return DrugIdentity(
        canonical_name=canonical_name,
        brand_names=brand_names,
        generic_names=generic_names,
        substance_names=substance_names,
        application_numbers=application_numbers,
        market_authorization_date=auth_date,
        market_authorization_date_source=auth_source,
        resolution_confidence=resolution_confidence,
    )


def _resolve_name_variants(raw_name: str, client: FDAClient):
    """Primary exact lookup against label.json, with broadened/ndc.json fallbacks.

    No custom fuzzy-matching library is used — misspelling tolerance is
    bounded by whatever leniency openFDA's own search grammar provides
    (unquoted/broadened term matching). This is a known, documented
    limitation: genuine typos (transposed/missing letters) may not resolve.
    Callers must treat a resulting DrugIdentityError(reason="NOT_FOUND") as
    an acceptable, correct outcome for such cases.
    """
    attempted_variants = []
    safe_name = client._sanitize_for_query(raw_name)

    attempted_variants.append(f"label.json exact brand/generic: {raw_name}")
    exact_search = f'(openfda.brand_name:"{safe_name}" OR openfda.generic_name:"{safe_name}")'
    records = _query_endpoint(client, LABEL_ENDPOINT, exact_search, DRUG_IDENTITY_LABEL_LIMIT)
    if records:
        return records, "EXACT", attempted_variants

    attempted_variants.append(f"label.json broadened brand/generic/substance: {raw_name}")
    broadened_search = (
        f'(openfda.brand_name:{safe_name} OR openfda.generic_name:{safe_name}'
        f' OR openfda.substance_name:{safe_name})'
    )
    records = _query_endpoint(client, LABEL_ENDPOINT, broadened_search, DRUG_IDENTITY_LABEL_LIMIT)
    if records:
        return records, "FUZZY", attempted_variants

    attempted_variants.append(f"ndc.json broadened brand/generic: {raw_name}")
    ndc_search = f'(brand_name:{safe_name} OR generic_name:{safe_name})'
    records = _query_endpoint(client, NDC_ENDPOINT, ndc_search, DRUG_IDENTITY_NDC_LIMIT)
    if records:
        return records, "FUZZY", attempted_variants

    return [], "EXACT", attempted_variants


def _query_endpoint(client: FDAClient, url: str, search: str, limit: int) -> list:
    response = client._resilient_get(
        url,
        params={"search": search, "limit": limit, "api_key": client.api_key},
        timeout=API_TIMEOUT_DEFAULT,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return response.json().get("results", [])


def _union_openfda_fields(records: list):
    """Union brand/generic/substance names and application numbers across all
    matched records. Handles both label.json's openfda-nested list fields and
    ndc.json's top-level singular brand_name/generic_name fields."""
    brand_names, generic_names, substance_names, application_numbers = [], [], [], []

    def _add_all(target: list, values) -> None:
        for v in values:
            if v and v not in target:
                target.append(v)

    for record in records:
        openfda = record.get("openfda") or {}
        _add_all(brand_names, openfda.get("brand_name", []))
        _add_all(generic_names, openfda.get("generic_name", []))
        _add_all(substance_names, openfda.get("substance_name", []))
        _add_all(application_numbers, openfda.get("application_number", []))

        # ndc.json records carry brand_name/generic_name as top-level singular
        # strings rather than list fields nested under openfda.
        if record.get("brand_name"):
            _add_all(brand_names, [record["brand_name"]])
        if record.get("generic_name"):
            _add_all(generic_names, [record["generic_name"]])

    return brand_names, generic_names, substance_names, application_numbers


def _lookup_market_authorization_date(application_numbers: list, brand_names: list,
                                      generic_names: list, client: FDAClient):
    """Earliest AP-status submission_status_date across all matched applications
    in drugsfda.json. Prefers exact application_number match; falls back to a
    brand/generic name search when no application_number was recoverable.

    Known limitations (documented per PSUR_CONSOLIDATION_SCOPE.md Section 7,
    not silently resolved):
    - A drug with multiple NDAs/ANDAs sharing a name (brand + several generic
      manufacturers) will have this pick the single earliest AP date across
      all of them, conflating "when was the first version of this substance
      approved by anyone" with "when was this specific product approved."
    - OTC monograph drugs (e.g. aspirin, ibuprofen) carry an OTC monograph
      number (e.g. "M013") rather than an NDA/ANDA/BLA and have no
      drugsfda.json entry at all — market_authorization_date is legitimately
      None for these, surfaced via resolution_confidence="PARTIAL" rather
      than treated as an error.
    """
    applications = []

    if application_numbers:
        for app_num in application_numbers:
            safe_app_num = client._sanitize_for_query(app_num)
            applications.extend(
                _query_drugsfda(client, f'application_number:"{safe_app_num}"'))
        source = f"drugsfda.json:{','.join(application_numbers)}"
    else:
        fallback_name = (brand_names[:1] or generic_names[:1] or [None])[0]
        if fallback_name is None:
            return None, "no application_number or brand/generic name available for drugsfda.json lookup"
        safe_name = client._sanitize_for_query(fallback_name)
        applications = _query_drugsfda(
            client,
            f'(openfda.brand_name:"{safe_name}" OR openfda.generic_name:"{safe_name}")',
        )
        source = f"drugsfda.json:name-fallback:{fallback_name}"

    earliest = _earliest_ap_date(applications)
    if earliest is None:
        return None, "no AP-status submission found in drugsfda.json"
    return earliest, source


def _query_drugsfda(client: FDAClient, search: str) -> list:
    response = client._resilient_get(
        DRUGSFDA_ENDPOINT,
        params={"search": search, "limit": DRUG_IDENTITY_DRUGSFDA_LIMIT, "api_key": client.api_key},
        timeout=API_TIMEOUT_DEFAULT,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return response.json().get("results", [])


def _earliest_ap_date(applications: list) -> Optional[date]:
    dates = []
    for app in applications:
        for submission in app.get("submissions", []):
            if submission.get("submission_status") != DRUGSFDA_APPROVED_STATUS:
                continue
            raw_date = submission.get("submission_status_date")
            if not raw_date:
                continue
            try:
                dates.append(datetime.strptime(raw_date, "%Y%m%d").date())
            except ValueError:
                continue
    return min(dates) if dates else None
