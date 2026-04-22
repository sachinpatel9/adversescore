import calendar
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from .config import (
    initialize_config,
    RETRY_TOTAL, RETRY_BACKOFF_FACTOR, RETRY_STATUS_CODES,
    TENACITY_MAX_ATTEMPTS, TENACITY_WAIT_MULTIPLIER, TENACITY_WAIT_MIN, TENACITY_WAIT_MAX,
    API_TIMEOUT_DEFAULT, API_TIMEOUT_AGGREGATION,
    DEFAULT_DAYS_BACK, DEFAULT_EVENT_LIMIT, DEFAULT_COUNT_LIMIT,
    AGE_COHORT_RANGE, MAX_PEERS, MIN_PEER_NAME_LENGTH, LABEL_FALLBACK_LIMIT,
)
from .logger import get_logger, log_event

logger = get_logger("fda")


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
                'company': report.get('companynumb', 'N/A')
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
