"""Consolidation orchestration — Phase 6 of the PSUR rebuild.

Wires the five already-independent Phase 1-5 pipeline stages into a single
callable entry point, `consolidate_psur()`, so later phases (persistence,
Phase 8's agent orchestration) have one clean function to call instead of
composing five modules themselves:

    1. Drug identity resolution      (drug_identity.resolve_drug_identity)
    2. Chunked, paginated FDA retrieval (FDAClient.fetch_psur_reports)
    3. Deduplication                 (deduplication.deduplicate_reports)
    4. Label status classification   (implicit — folded into ranking via
       label_classifier.classify_label_statuses, called from ranking.rank_signals)
    5. Deterministic ranking         (ranking.rank_signals)

This module makes no HTTP calls of its own — all HTTP stays inside
`FDAClient` / `drug_identity.py`. It is pure wiring: unpack a resolved
`DrugIdentity` into the plain primitives `FDAClient.fetch_psur_reports`
expects, thread outputs from one stage into the next, and compose the final
result.

Convention, consistent with every other Phase 1-5 module: this module never
raises for a domain-level failure (unresolvable drug name, missing API keys,
no queryable name variants). Each such case returns a structured
`ConsolidationError` describing what stage failed and why. Only truly
unexpected exceptions (bugs, not domain failures) propagate as real
exceptions — the same "structured result, never raise" contract used by
`drug_identity.py`'s `DrugIdentityError` and `fda_client.py`'s
`PSURRetrievalResult`/`ChunkResult` degraded-result pattern.
"""
from dataclasses import dataclass
from typing import Literal, Optional, Union

from .deduplication import deduplicate_reports, DedupResult
from .drug_identity import resolve_drug_identity, DrugIdentity, DrugIdentityError
from .fda_client import FDAClient, PSURRetrievalResult
from .logger import get_logger, log_event
from .ranking import rank_signals, RankingResult

logger = get_logger("consolidation")


# ── Output Data Model ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConsolidationResult:
    drug_identity: DrugIdentity
    period: str                        # "6mo" | "1yr" | "2yr" | "3yr"
    retrieval: PSURRetrievalResult
    dedup: DedupResult
    ranking: RankingResult
    pharm_class: str                   # "" if discovery failed/found nothing
    class_counts_available: bool       # False when pharm_class discovery yielded no background counts
    formula_version: str               # passthrough of ranking.formula_version


@dataclass(frozen=True)
class ConsolidationError:
    drug_name: str
    stage: Literal["CLIENT_CONSTRUCTION", "IDENTITY_RESOLUTION", "RETRIEVAL"]
    reason: str
    message: str


# ── Entry Point ────────────────────────────────────────────────────────────

def consolidate_psur(
    drug_name: str,
    period: Literal["6mo", "1yr", "2yr", "3yr"],
    client: Optional[FDAClient] = None,
) -> Union[ConsolidationResult, ConsolidationError]:
    """Run the full PSUR consolidation pipeline for `drug_name` over `period`.

    Dependency-injection pattern mirrors `resolve_drug_identity` — pass an
    existing `FDAClient` (e.g. a session-scoped test fixture) or leave `client`
    None to construct one. Depends on `FDAClient` directly rather than
    `AdverseScoreClient`: `fetch_psur_reports` only exists on `FDAClient`, and
    `AdverseScoreClient` (a thin delegation wrapper) adds no value here.

    Known limitation: label text lookup (`fetch_label_text`) is queried using
    only `identity.canonical_name` — no fallback loop through other name
    variants. A generic-only drug with no `brand_names` may get an empty
    label lookup, since `fetch_label_text` queries openFDA's `brand_name`
    field specifically. This degrades gracefully downstream: `rank_signals`
    (via `classify_label_statuses`) treats an empty label as
    `LABEL_STATUS_UNKNOWN` for every symptom, never a crash.

    Never raises for a domain-level failure — returns a structured
    `ConsolidationError` instead. See module docstring for the full
    "structured result, never raise" convention.
    """
    if client is None:
        try:
            client = FDAClient()
        except EnvironmentError as e:
            log_event(logger, "consolidation_client_construction_failed", drug=drug_name, error=str(e))
            return ConsolidationError(
                stage="CLIENT_CONSTRUCTION",
                reason="MISSING_API_KEYS",
                drug_name=drug_name,
                message=str(e),
            )

    identity = resolve_drug_identity(drug_name, client=client)
    if isinstance(identity, DrugIdentityError):
        return ConsolidationError(
            stage="IDENTITY_RESOLUTION",
            reason=identity.reason,
            drug_name=drug_name,
            message=identity.message,
        )

    name_variants = _build_name_variants(identity)
    if not name_variants:
        # Defensive guard — shouldn't normally happen given how canonical_name
        # is derived from brand_names/generic_names, but fetch_psur_reports
        # raises ValueError on an empty list, and we want a structured error
        # here instead of letting that propagate.
        return ConsolidationError(
            stage="RETRIEVAL",
            reason="NO_NAME_VARIANTS",
            drug_name=drug_name,
            message=f"No queryable brand/generic/substance name variants resolved for '{drug_name}'.",
        )

    retrieval = client.fetch_psur_reports(
        name_variants, identity.market_authorization_date, period)

    dedup = deduplicate_reports(retrieval.reports)

    label_text = client.fetch_label_text(identity.canonical_name)

    pharm_class = client._discover_drug_class(identity.canonical_name)
    if pharm_class:
        class_counts = client._fetch_symptom_counts(
            pharm_class=pharm_class,
            start_date=retrieval.period_start,
            end_date=retrieval.period_end,
        )
    else:
        class_counts = {}
    class_counts_available = bool(class_counts)

    ranking = rank_signals(dedup.reports, class_counts, label_text)

    log_event(logger, "consolidation_complete", drug=drug_name, period=period,
              total_retrieved=retrieval.total_retrieved_count,
              total_deduped=dedup.total_output_count,
              total_signals=ranking.total_signals,
              class_counts_available=class_counts_available)

    return ConsolidationResult(
        drug_identity=identity,
        period=period,
        retrieval=retrieval,
        dedup=dedup,
        ranking=ranking,
        pharm_class=pharm_class,
        class_counts_available=class_counts_available,
        formula_version=ranking.formula_version,
    )


def _build_name_variants(identity: DrugIdentity) -> list:
    """Deduped (order-preserving, case-sensitive), non-empty union of
    brand/generic/substance name variants from a resolved DrugIdentity."""
    variants = []
    for name in identity.brand_names + identity.generic_names + identity.substance_names:
        if name and name.strip() and name not in variants:
            variants.append(name)
    return variants
