import calendar
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from typing import Optional
from .config import initialize_config


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
            total=3,
            backoff_factor=1,
            status_forcelist=[429,500,502,503,504],
            allowed_methods=['GET']
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)
        return session

    def _sanitize_for_query(self, value: str) -> str:
        # Drug names and class names are interpolated into Lucene quoted strings
        return value.replace('\\', '\\\\').replace('"', '\\"')

    def build_query(self, drug_name: str, days_back: int = 365, limit: int = 500,
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
            lower_bound = max(0, patient_age - 5)
            upper_bound = patient_age + 5
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
            response = self.session.get(full_url, timeout=10)
            if response.status_code == 404:
                print(f"[API] No adverse event reports found for {drug_name} in the queried time range.")
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[API] HTTP error fetching data for {drug_name}: {e}")
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
            response = self.session.get(f"{label_url}?{query}&api_key={self.api_key}", timeout=10)
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
            print(f"[Discovery] Attempting algorithmic class mapping for {target_name}...")
            response = self.session.get(
                url,
                params={"search": search_str, "count": count_param, "api_key": self.api_key},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            if results:
                primary_class = results[0].get('term')
                count = results[0].get('count')
                print(f"[Discovery] Class Identified: {primary_class} (N={count})")
                return primary_class

            return ""
        except Exception as e:
            print(f"[Discovery] Event based discovery failed: {str(e)}. Falling back to label metadata...")
            return self._fetch_label_class_fallback(target_name)

    def _fetch_label_class_fallback(self, drug_name: str) -> str:
        '''
        Helper to ensure we do not return any empty string if the event API is noisy.
        '''
        url = "https://api.fda.gov/drug/label.json"
        safe_name = self._sanitize_for_query(drug_name)
        search_value = f'openfda.brand_name:"{safe_name}"'

        try:
            response = self.session.get(
                url,
                params={"search": search_value, "limit": 5, "api_key": self.api_key},
                timeout=10,
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

        print(f"[Discovery] Pharmacologic Class Identified: {pharm_class}")
        url = "https://api.fda.gov/drug/event.json"
        clean_class = self._sanitize_for_query(pharm_class).replace(' ', '+')
        query = f'search=patient.drug.openfda.pharm_class_epc:"{clean_class}"&count=patient.drug.medicinalproduct.exact'

        try:
            response = self.session.get(f"{url}?{query}&api_key={self.api_key}", timeout=10)
            response.raise_for_status()
            data = response.json()

            peers = []
            target_upper = target_drug.upper()

            for item in data.get('results', []):
                peer_name = item.get('term', '').upper()
                if peer_name and peer_name != target_upper and len(peer_name) > 3:
                    peers.append(peer_name)
                if len(peers) >= 3:
                    break

            print(f"[Discovery] Top Peers Found: {', '.join(peers)}")
            return peers

        except Exception as e:
            print(f"[Discovery] Failed to find peers for class {pharm_class}.")
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
            lower = max(0, patient_age - 5)
            upper = patient_age + 5
            query_parts.append(f'patient.patientonsetage:[{lower}+TO+{upper}]')
        if start_date and end_date:
            query_parts.append(f'receivedate:[{start_date}+TO+{end_date}]')

        search_string = "+AND+".join([q.replace(" ", "+") for q in query_parts])
        url = f"{self.base_url}?search={search_string}&count=patient.reaction.reactionmeddrapt.exact&limit=1000&api_key={self.api_key}"

        try:
            entity = drug_name or pharm_class
            print(f"[Aggregation] Running server-side symptom count for {entity}")
            response = self.session.get(url, timeout=15)
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
