import calendar
import logging
import requests
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import date, datetime, timedelta
from typing import Literal, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from .config import (
    initialize_config,
    RETRY_TOTAL, RETRY_BACKOFF_FACTOR, RETRY_STATUS_CODES,
    TENACITY_MAX_ATTEMPTS, TENACITY_WAIT_MULTIPLIER, TENACITY_WAIT_MIN, TENACITY_WAIT_MAX,
    API_TIMEOUT_DEFAULT, API_TIMEOUT_AGGREGATION,
    DEFAULT_DAYS_BACK, DEFAULT_EVENT_LIMIT, DEFAULT_COUNT_LIMIT,
    AGE_COHORT_RANGE, MAX_PEERS, MIN_PEER_NAME_LENGTH, LABEL_FALLBACK_LIMIT,
    PSUR_PAGE_SIZE, PSUR_SKIP_CEILING, PSUR_CHUNK_MONTHS,
    PSUR_PERIOD_MONTHS, PSUR_PERIOD_FALLBACK_DAYS,
)
from .logger import get_logger, log_event

logger = get_logger("fda")


# ── PSUR Chunked Retrieval — Output Data Model (Phase 2) ─────────────────────

@dataclass(frozen=True)
class ChunkResult:
    label: str
    start_date: str              # YYYYMMDD
    end_date: str                # YYYYMMDD
    reports: list                # flattened report dicts (_flatten_results shape)
    retrieved_count: int
    estimated_total_count: int
    truncated: bool
    error: Optional[str] = None  # non-None only if a page failure aborted this chunk early


@dataclass(frozen=True)
class PSURRetrievalResult:
    canonical_query_variants: list   # name_variants actually queried, for traceability
    period_start: str                # YYYYMMDD
    period_end: str                  # YYYYMMDD
    chunks: list                     # list[ChunkResult], chronological order
    reports: list                    # merged, boundary-deduplicated flattened report dicts
    total_retrieved_count: int       # len(reports) post-merge
    total_estimated_count: int       # sum of per-chunk estimated_total_count
    any_chunk_truncated: bool
    used_fallback_anchor: bool
    fallback_reason: Optional[str] = None  # "NO_MARKET_AUTH_DATE" | "AUTH_DATE_LESS_THAN_ONE_PERIOD_AGO" | None


# ── PSUR Period/Chunk Date Math (Phase 2) — pure functions, no HTTP ──────────

def _add_months(d: date, months: int) -> date:
    """Stdlib-only month arithmetic with day-of-month clamping (e.g. Jan 31 + 1mo -> Feb 28/29).
    No new dependency — python-dateutil is only a transitive/undeclared dep via pandas, not
    used anywhere in src/adverse_score/ today."""
    total_month_index = d.month - 1 + months
    year = d.year + total_month_index // 12
    month = total_month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def compute_psur_period(
    market_authorization_date: Optional[date],
    period: Literal["6mo", "1yr", "2yr", "3yr"],
    as_of: Optional[date] = None,
) -> tuple:
    """Compute the PSUR reporting date range, anchored to the drug's market authorization
    date (International Birth Date) per real ICH E2C(R2) PBRER practice — a fixed-length
    cycle counted forward from the anchor, selecting the most recently *completed* cycle as
    of `as_of` (defaults to today).

    Returns (period_start: date, period_end: date, used_fallback_anchor: bool,
    fallback_reason: Optional[str]).

    Falls back to a simple rolling lookback from `as_of` (period_end=as_of,
    period_start=as_of - PSUR_PERIOD_FALLBACK_DAYS[period]) when market_authorization_date
    is None (e.g. OTC monograph drugs with no NDA/ANDA/BLA record — see drug_identity.py)
    or when the drug was approved less than one full period ago (no completed cycle exists
    yet). Both cases are flagged via fallback_reason so callers can surface the caveat
    rather than silently treating the range as anchor-derived.
    """
    if as_of is None:
        as_of = date.today()

    months = PSUR_PERIOD_MONTHS[period]

    if market_authorization_date is not None:
        n = 0
        while _add_months(market_authorization_date, months * (n + 1)) <= as_of:
            n += 1
        if n >= 1:
            period_end = _add_months(market_authorization_date, months * n)
            period_start = _add_months(market_authorization_date, months * (n - 1))
            return period_start, period_end, False, None
        fallback_reason = "AUTH_DATE_LESS_THAN_ONE_PERIOD_AGO"
    else:
        fallback_reason = "NO_MARKET_AUTH_DATE"

    period_end = as_of
    period_start = as_of - timedelta(days=PSUR_PERIOD_FALLBACK_DAYS[period])
    return period_start, period_end, True, fallback_reason


def _compute_psur_chunks(period_start: date, period_end: date,
                         chunk_months: int = PSUR_CHUNK_MONTHS) -> list:
    """Walk forward from period_start in chunk_months-month steps, clamping the final chunk
    to period_end. Returns [(label, start_YYYYMMDD, end_YYYYMMDD), ...] — same 3-tuple shape
    as the old _compute_quarter_boundaries for stylistic consistency, but NOT calendar-
    quarter-aligned (chunks are counted from the PSUR period's anchor date, per IBD-cycle
    anchoring), so labels use "chunk-{i}-{start}-to-{end}" rather than "YYYY-Qn" (which would
    be misleading here). Do not modify the old _compute_quarter_boundaries — it is unrelated
    and still calendar-anchored-to-now for its own callers.

    Chunk date ranges intentionally overlap by one day at boundaries: openFDA's
    receivedate:[X TO Y] range query is inclusive on both ends, and FAERS receivedate has
    day granularity only, so a report received exactly on a boundary day can be returned by
    both adjacent chunks' queries. Making ranges exclusive would require fragile
    month-boundary-aware date subtraction; instead the one-day overlap is handled by
    merge-time dedup (_merge_chunks), not by "fixing" the boundaries here.
    """
    chunks = []
    i = 0
    chunk_start = period_start
    while chunk_start < period_end:
        chunk_end = min(_add_months(period_start, chunk_months * (i + 1)), period_end)
        label = f"chunk-{i + 1}-{chunk_start.strftime('%Y%m%d')}-to-{chunk_end.strftime('%Y%m%d')}"
        chunks.append((label, chunk_start.strftime('%Y%m%d'), chunk_end.strftime('%Y%m%d')))
        i += 1
        chunk_start = chunk_end
    return chunks


def _extract_drug_names(report: dict) -> list:
    """Pulls patient.drug[].medicinalproduct for every drug on the case (suspect,
    concomitant, and interacting alike — drug-name matching elsewhere in this file
    is already characterization-agnostic, so this stays consistent). Filters blanks,
    dedupes while preserving order."""
    raw_drugs = report.get('patient', {}).get('drug') or []
    names = []
    seen = set()
    for d in raw_drugs:
        name = d.get('medicinalproduct')
        if name and name.strip() and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _parse_version(raw) -> int:
    """Parses safetyreportversion to int, defaulting to 1 if missing/unparseable
    (a genuinely missing version is treated as 'version 1' — documented limitation,
    there's no better fallback without the field)."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


def _merge_chunks(chunk_results: list) -> list:
    """First-seen-wins merge on report_id across chunks, in chronological chunk order.

    This is boundary-collision list hygiene only — satisfying Phase 2's own "no duplicate
    reports in the merged output" acceptance criterion when the same safetyreportid is
    returned by two adjacent chunks' overlapping date ranges (see _compute_psur_chunks). It
    is explicitly NOT Phase 3's deduplication pipeline: Phase 3 owns fuzzy/near-duplicate
    content matching and audited dedup statistics across reports that have genuinely
    different safetyreportids but represent the same underlying case. Do not extend this
    function with additional dedup heuristics — that belongs in Phase 3.

    Version-aware: if the same report_id recurs with a higher safetyreportversion
    (FAERS case amendment), the later/higher-version payload replaces the earlier one
    in place — this is still exact-ID collision resolution, not Phase 3's fuzzy dedup.
    """
    best_by_id = {}
    order = []
    for chunk in chunk_results:
        for report in chunk.reports:
            rid = report.get("report_id")
            existing = best_by_id.get(rid)
            if existing is None:
                best_by_id[rid] = report
                order.append(rid)
            elif report.get("safetyreportversion", 1) > existing.get("safetyreportversion", 1):
                best_by_id[rid] = report
    return [best_by_id[rid] for rid in order]


class FDAClient:
    """Handles all HTTP communication with the openFDA API."""

    base_url = "https://api.fda.gov/drug/event.json"

    def __init__(self):
        self.api_key: str = initialize_config()
        self.session = self._get_transport_session()

    def _get_transport_session(self):
        '''
        Creates a session with automated retry logic
        handles 429 (rate limit), 500, 502, 503, 504 errors
        '''
        session = requests.Session()
        retries = Retry(
            total=RETRY_TOTAL,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=['GET']
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)
        return session

    @retry(
        stop=stop_after_attempt(TENACITY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=TENACITY_WAIT_MULTIPLIER, min=TENACITY_WAIT_MIN, max=TENACITY_WAIT_MAX),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _resilient_get(self, *args, **kwargs):
        """session.get with tenacity retry for transient failures."""
        return self.session.get(*args, **kwargs)

    def _sanitize_for_query(self, value: str) -> str:
        # Drug names and class names are interpolated into Lucene quoted strings
        return value.replace('\\', '\\\\').replace('"', '\\"')

    def build_query(self, drug_name: str, days_back: int = DEFAULT_DAYS_BACK, limit: int = DEFAULT_EVENT_LIMIT,
                    patient_age: Optional[int] = None, patient_sex: Optional[str] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        '''
        Constructs a valid openFDA Lucene search query.
        Example output: search=patient.drug.medicinalproduct:'TYLENOL'+AND+receivedate:[20231210+TO+20240310]
        '''
        if start_date and end_date:
            date_range = f"[{start_date}+TO+{end_date}]"
        else:
            end_date_str = datetime.now().strftime('%Y%m%d')
            start_date_str = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            date_range = f"[{start_date_str}+TO+{end_date_str}]"

        safe_name = self._sanitize_for_query(drug_name)
        search_params = f'patient.drug.medicinalproduct:"{safe_name}" AND receivedate:{date_range}'

        if patient_sex:
            sex_code = "2" if patient_sex.upper() == "F" else "1"
            search_params += f' AND patient.patientsex:{sex_code}'

        if patient_age:
            lower_bound = max(0, patient_age - AGE_COHORT_RANGE)
            upper_bound = patient_age + AGE_COHORT_RANGE
            search_params += f' AND patient.patientonsetage:[{lower_bound}+TO+{upper_bound}]'

        encoded_search = search_params.replace(" ", "+")
        return f"search={encoded_search}&limit={limit}"

    def fetch_events(self, drug_name: str, patient_age: int = None, patient_sex: str = None,
                     start_date: str = None, end_date: str = None):  # type: ignore
        '''
        Executes the API call using the query builder and the session.
        Note on pagination: openFDA caps results at limit=1000 and skip+limit<=26000.
        We fetch 500 reports as a representative sample. The downstream confidence metric
        accounts for sample size, and the count endpoints used for PRR aggregate server-side
        with no pagination cap. Fetching all reports is not practical for real-time scoring.
        '''
        query_params = self.build_query(drug_name, patient_age=patient_age,
                                        patient_sex=patient_sex, start_date=start_date,
                                        end_date=end_date)
        full_url = f"{self.base_url}?{query_params}&api_key={self.api_key}"

        try:
            response = self._resilient_get(full_url, timeout=API_TIMEOUT_DEFAULT)
            if response.status_code == 404:
                log_event(logger, "fetch_events_empty", drug=drug_name)
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log_event(logger, "fetch_events_error", drug=drug_name, error=str(e))
            return None

    def _flatten_results(self, raw_data) -> list:
        '''
        Transforms messy FDA JSON into a flat, Agent-friendly list of dictionaries
        '''
        if not raw_data or 'results' not in raw_data:
            return []
        flattened = []
        for report in raw_data.get('results', []):
            raw_reactions = report.get('patient', {}).get('reaction') or []
            reactions = [r.get('reactionmeddrapt', 'Unknown') for r in raw_reactions]

            entry = {
                'report_id': report.get('safetyreportid'),
                'date': report.get('receivedate'),
                'severity': 'Serious' if report.get('seriousness') == '1' else 'Non-Serious',
                'is_death': report.get('seriousnessdeath') == '1',
                'is_hospitalization': report.get('seriousnesshospitalization') == '1',
                'symptoms': ", ".join(reactions),
                'company': report.get('companynumb', 'N/A'),
                'symptom_list': reactions,
                'drug_names': _extract_drug_names(report),
                'safetyreportversion': _parse_version(report.get('safetyreportversion')),
            }
            flattened.append(entry)
        return flattened

    def fetch_label_text(self, drug_name: str) -> str:
        '''
        Retrieves official FDA 'Adverse Reactions' text
        Used to identify 'Unlabeled vs Labeled signals
        '''
        label_url = 'https://api.fda.gov/drug/label.json'
        safe_name = self._sanitize_for_query(drug_name)
        query = f'search=openfda.brand_name:"{safe_name}"&limit=1'

        try:
            response = self._resilient_get(f"{label_url}?{query}&api_key={self.api_key}", timeout=API_TIMEOUT_DEFAULT)
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            if results:
                reactions_section = results[0].get('adverse_reactions', [])
                return " ".join(reactions_section).lower()
            return ""
        except Exception:
            return ""

    def _discover_drug_class(self, drug_name: str) -> str:
        '''
        Discovers the primary therapeutic class using frequency analysis on historical adverse event data, bypassing label noise
        '''
        url = "https://api.fda.gov/drug/event.json"
        target_name = drug_name.upper()
        safe_name = self._sanitize_for_query(target_name)
        search_str = f'patient.drug.openfda.brand_name:"{safe_name}"'
        count_param = "patient.drug.openfda.pharm_class_epc.exact"

        try:
            log_event(logger, "discover_class_start", drug=target_name)
            response = self._resilient_get(
                url,
                params={"search": search_str, "count": count_param, "api_key": self.api_key},
                timeout=API_TIMEOUT_DEFAULT,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            if results:
                primary_class = results[0].get('term')
                count = results[0].get('count')
                log_event(logger, "discover_class_found", drug=target_name, pharm_class=primary_class, count=count)
                return primary_class

            return ""
        except Exception as e:
            log_event(logger, "discover_class_fallback", drug=target_name, error=str(e))
            return self._fetch_label_class_fallback(target_name)

    def _fetch_label_class_fallback(self, drug_name: str) -> str:
        '''
        Helper to ensure we do not return any empty string if the event API is noisy.
        '''
        url = "https://api.fda.gov/drug/label.json"
        safe_name = self._sanitize_for_query(drug_name)
        search_value = f'openfda.brand_name:"{safe_name}"'

        try:
            response = self._resilient_get(
                url,
                params={"search": search_value, "limit": LABEL_FALLBACK_LIMIT, "api_key": self.api_key},
                timeout=API_TIMEOUT_DEFAULT,
            )
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            res = response.json()

            ignore_classes = ["Endoglycosidase [EPC]", "Hyaluronidase"]

            for result in res.get('results', []):
                epc_list = result.get('openfda', {}).get('pharm_class_epc', [])
                for epc in epc_list:
                    if epc not in ignore_classes:
                        return epc
            return ""
        except Exception:
            return ""

    def _discover_peers(self, pharm_class: str, target_drug: str) -> list:  # type: ignore
        '''
        Find the top 3 most prescribed / reported peer drugs in the same pharmacologic class.
        Uses the openFDA event count endpoint
        '''
        if not pharm_class:
            return []

        log_event(logger, "discover_peers_start", pharm_class=pharm_class)
        url = "https://api.fda.gov/drug/event.json"
        clean_class = self._sanitize_for_query(pharm_class).replace(' ', '+')
        query = f'search=patient.drug.openfda.pharm_class_epc:"{clean_class}"&count=patient.drug.medicinalproduct.exact'

        try:
            response = self._resilient_get(f"{url}?{query}&api_key={self.api_key}", timeout=API_TIMEOUT_DEFAULT)
            response.raise_for_status()
            data = response.json()

            peers = []
            target_upper = target_drug.upper()

            for item in data.get('results', []):
                peer_name = item.get('term', '').upper()
                if peer_name and peer_name != target_upper and len(peer_name) > MIN_PEER_NAME_LENGTH:
                    peers.append(peer_name)
                if len(peers) >= MAX_PEERS:
                    break

            log_event(logger, "discover_peers_found", peers=peers)
            return peers

        except Exception as e:
            log_event(logger, "discover_peers_error", pharm_class=pharm_class)
            return []

    def _fetch_symptom_counts(self, drug_name: str = None, pharm_class: str = None,
                              patient_age: int = None, patient_sex: str = None,
                              start_date: str = None, end_date: str = None) -> dict:  # type: ignore
        '''
        Hits the openFDA count endpoint for a specific drug or a pharmacologic class - forces the FDA servers to aggregate symptom frequencies instantly
        '''
        query_parts = []

        if drug_name:
            safe_name = self._sanitize_for_query(drug_name)
            query_parts.append(f'patient.drug.medicinalproduct:"{safe_name}"')
        elif pharm_class:
            clean_class = pharm_class.replace(' ', '+').replace('"', '')
            query_parts.append(f'patient.drug.openfda.pharm_class_epc:"{clean_class}"')

        if patient_sex:
            sex_code = "2" if patient_sex.upper() == "F" else "1"
            query_parts.append(f'patient.patientsex:{sex_code}')
        if patient_age:
            lower = max(0, patient_age - AGE_COHORT_RANGE)
            upper = patient_age + AGE_COHORT_RANGE
            query_parts.append(f'patient.patientonsetage:[{lower}+TO+{upper}]')
        if start_date and end_date:
            query_parts.append(f'receivedate:[{start_date}+TO+{end_date}]')

        search_string = "+AND+".join([q.replace(" ", "+") for q in query_parts])
        url = f"{self.base_url}?search={search_string}&count=patient.reaction.reactionmeddrapt.exact&limit={DEFAULT_COUNT_LIMIT}&api_key={self.api_key}"

        try:
            entity = drug_name or pharm_class
            log_event(logger, "symptom_count_start", entity=entity)
            response = self._resilient_get(url, timeout=API_TIMEOUT_AGGREGATION)
            response.raise_for_status()
            data = response.json()
            counts = {}
            for item in data.get('results', []):
                term = item.get('term')
                count = item.get('count')
                if term is not None and count is not None:
                    counts[term.upper()] = count
            return counts
        except Exception:
            return {}

    def _compute_quarter_boundaries(self, num_quarters: int = 4) -> list:
        """Return [(label, start_YYYYMMDD, end_YYYYMMDD), ...] for the last N calendar quarters."""
        today = datetime.now()
        current_q = (today.month - 1) // 3
        current_year = today.year
        quarters = []
        for i in range(num_quarters - 1, -1, -1):
            q_index = current_q - i
            year = current_year
            while q_index < 0:
                q_index += 4
                year -= 1
            month_start = q_index * 3 + 1
            month_end = (q_index + 1) * 3 if q_index < 3 else 12
            month_end_day = calendar.monthrange(year, month_end)[1]
            label = f"{year}-Q{q_index + 1}"
            start = f"{year}{month_start:02d}01"
            end = f"{year}{month_end:02d}{month_end_day:02d}"
            quarters.append((label, start, end))
        return quarters

    # ── PSUR Chunked, Paginated Retrieval (Phase 2) ──────────────────────────
    # fda_client.py must not import drug_identity.DrugIdentity — drug_identity.py already
    # imports FDAClient, so the reverse would risk a circular import. All inputs below are
    # plain primitives (name variant strings, an Optional[date] anchor, a period string);
    # Phase 6's consolidation.py is responsible for unpacking a resolved DrugIdentity into
    # these primitives before calling fetch_psur_reports().

    def _build_psur_chunk_query(self, name_variants: list, start_date: str, end_date: str) -> dict:
        """Returns a params dict (not a raw URL string) — matches _discover_drug_class's and
        drug_identity.py's transport style.

        Critical: uses literal spaces around OR/AND/TO, not the +OR+/+AND+ literal-plus style
        used in fetch_events'/_fetch_symptom_counts' raw-URL-string code paths. This query is
        submitted via params=, so requests handles all encoding — embedding literal "+" text
        double-encodes to %2B and openFDA's server returns 500s (the exact bug class fixed in
        Phase 1's drug_identity.py after a live E2E failure). Do not copy the +-based style
        from fetch_events into this method.
        """
        deduped = list(dict.fromkeys(v.upper() for v in name_variants if v))
        safe_variants = [self._sanitize_for_query(v) for v in deduped]
        or_clause = "(" + " OR ".join(f'patient.drug.medicinalproduct:"{v}"' for v in safe_variants) + ")"
        search = f'{or_clause} AND receivedate:[{start_date} TO {end_date}]'
        return {"search": search, "api_key": self.api_key}

    def _fetch_chunk_paginated(self, name_variants: list, label: str,
                               start_date: str, end_date: str) -> ChunkResult:
        """Paginate a single date-bounded chunk with skip/limit, staying under
        PSUR_SKIP_CEILING. Three termination cases: (1) a page returns fewer than
        PSUR_PAGE_SIZE results -> exhausted, not truncated; (2) the next skip would exceed
        PSUR_SKIP_CEILING -> truncated; (3) a 404 on the first page -> zero results, valid
        terminal state, not an error.

        A page failure (after _resilient_get's retries exhaust) aborts only this chunk, not
        the whole retrieval — the partial results collected so far are still returned, with
        `error` and `truncated` set. One flaky page in a multi-chunk retrieval should not
        discard otherwise-good data from other chunks, matching the codebase's existing
        philosophy of returning structured degraded results rather than raising.
        """
        skip = 0
        all_raw_results = []
        estimated_total = 0
        truncated = False
        page_error = None

        while True:
            params = self._build_psur_chunk_query(name_variants, start_date, end_date)
            params["limit"] = PSUR_PAGE_SIZE
            params["skip"] = skip

            try:
                response = self._resilient_get(self.base_url, params=params, timeout=API_TIMEOUT_DEFAULT)
                if response.status_code == 404:
                    break
                response.raise_for_status()
                page = response.json()
            except requests.exceptions.RequestException as e:
                page_error = str(e)
                truncated = True
                log_event(logger, "psur_chunk_page_error", label=label, skip=skip, error=page_error)
                break

            if skip == 0:
                estimated_total = page.get("meta", {}).get("results", {}).get("total", 0)

            page_results = page.get("results", [])
            all_raw_results.extend(page_results)

            if len(page_results) < PSUR_PAGE_SIZE:
                break

            skip += PSUR_PAGE_SIZE
            if skip > PSUR_SKIP_CEILING:
                truncated = True
                break

        reports = self._flatten_results({"results": all_raw_results})
        if not truncated and len(reports) < estimated_total:
            # Safety-net: any undercount not already caught by the ceiling check is still
            # surfaced as truncated rather than silently under-reporting completeness.
            truncated = True

        return ChunkResult(
            label=label,
            start_date=start_date,
            end_date=end_date,
            reports=reports,
            retrieved_count=len(reports),
            estimated_total_count=estimated_total,
            truncated=truncated,
            error=page_error,
        )

    def fetch_psur_reports(
        self,
        name_variants: list,
        market_authorization_date: Optional[date],
        period: Literal["6mo", "1yr", "2yr", "3yr"],
        as_of: Optional[date] = None,
    ) -> PSURRetrievalResult:
        """Retrieve all FAERS reports for a resolved drug identity across a PSUR reporting
        period, anchored to market_authorization_date, safely within openFDA's pagination
        limits. Public entry point for Phase 6's consolidation.py (not wired in this phase).
        """
        if not name_variants:
            raise ValueError("fetch_psur_reports requires at least one name variant")

        period_start, period_end, used_fallback_anchor, fallback_reason = compute_psur_period(
            market_authorization_date, period, as_of=as_of)

        chunk_bounds = _compute_psur_chunks(period_start, period_end)

        chunk_results = [
            self._fetch_chunk_paginated(name_variants, label, start_date, end_date)
            for label, start_date, end_date in chunk_bounds
        ]

        merged_reports = _merge_chunks(chunk_results)

        return PSURRetrievalResult(
            canonical_query_variants=list(dict.fromkeys(v.upper() for v in name_variants if v)),
            period_start=period_start.strftime('%Y%m%d'),
            period_end=period_end.strftime('%Y%m%d'),
            chunks=chunk_results,
            reports=merged_reports,
            total_retrieved_count=len(merged_reports),
            total_estimated_count=sum(c.estimated_total_count for c in chunk_results),
            any_chunk_truncated=any(c.truncated for c in chunk_results),
            used_fallback_anchor=used_fallback_anchor,
            fallback_reason=fallback_reason,
        )
