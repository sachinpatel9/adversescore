import math
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from .config import initialize_config
from typing import Optional, List, Dict
import json



class AdverseScoreClient:
    base_url = "https://api.fda.gov/drug/event.json"

    #Class Attributes 
    SEVERITY_WEIGHTS = {
        'DEATH': 1.75,
        'HOSPITALIZATION': 1.0,
        'OTHER_SERIOUS': 0.75,
        'NON_SERIOUS': 0.25
    }

    def __init__(self):
        self.api_key: str = initialize_config()
        self.session = self._get_transport_session()


    def build_query(self, drug_name: str, days_back: int = 365, limit: int = 500, patient_age: int = None, patient_sex: str = None, start_date: str = None, end_date: str = None) -> str:  # type: ignore
        '''
        Constructs a valid openFDA Lucene search query.
        Example output: search=patient.drug.medicinalproduct:'TYLENOL'+AND+receivedate:[20231210+TO+20240310]
        '''

        #Handle dates: YYYYMMDD format
        if start_date and end_date:
            date_range = f"[{start_date}+TO+{end_date}]"
        else:
            end_date_str = datetime.now().strftime('%Y%m%d')
            start_date_str = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
            date_range = f"[{start_date_str}+TO+{end_date_str}]"

        # Sanitize drug_name before embedding in Lucene query — an unescaped quote
        # in the name (e.g. a malformed LLM extraction) would break the query syntax.
        safe_name = self._sanitize_for_query(drug_name)
        #building the search parameters
        search_params = f'patient.drug.medicinalproduct:"{safe_name}" AND receivedate:{date_range}'

        #Inject Demographic Filters
        if patient_sex:
            sex_code = "2" if patient_sex.upper() == "F" else "1"
            search_params += f' AND patient.patientsex:{sex_code}'

        if patient_age:
            #create a 10 year age cohort bracket to ensure adequate sample size
            lower_bound = max(0, patient_age - 5)
            upper_bound = patient_age + 5
            search_params += f' AND patient.patientonsetage:[{lower_bound}+TO+{upper_bound}]'
        
        encoded_search = search_params.replace(" ", "+")

        return f"search={encoded_search}&limit={limit}"

    def _get_transport_session(self):
        '''
        Creates a session with automated retry logic
        handles 429 (rate limit), 500, 502, 503, 504 errors
        '''
        session = requests.Session()
        #configure retries
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429,500,502,503,504],
            allowed_methods=['GET']
        )

        #mount the adapter to the session
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)

        return session

    def _sanitize_for_query(self, value: str) -> str:
        # Drug names and class names are interpolated into Lucene quoted strings
        return value.replace('\\', '\\\\').replace('"', '\\"')

    def _compute_quarter_boundaries(self, num_quarters: int = 4) -> list:
        """Return [(label, start_YYYYMMDD, end_YYYYMMDD), ...] for the last N calendar quarters."""
        today = datetime.now()
        current_q = (today.month - 1) // 3  # 0-indexed: 0=Q1, 1=Q2, 2=Q3, 3=Q4
        current_year = today.year
        quarters = []
        for i in range(num_quarters - 1, -1, -1):
            q_index = current_q - i
            year = current_year
            while q_index < 0:
                q_index += 4
                year -= 1
            month_start = q_index * 3 + 1
            if q_index == 3:
                month_end = 12
                month_end_day = 31
            else:
                month_end = (q_index + 1) * 3
                if month_end in (1, 3, 5, 7, 8, 10, 12):
                    month_end_day = 31
                elif month_end == 2:
                    month_end_day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
                else:
                    month_end_day = 30
            label = f"{year}-Q{q_index + 1}"
            start = f"{year}{month_start:02d}01"
            end = f"{year}{month_end:02d}{month_end_day:02d}"
            quarters.append((label, start, end))
        return quarters

    def fetch_quarterly_data(self, drug_name, num_quarters=4, patient_age=None, patient_sex=None, target_symptom=None):
        """Calculate AdverseScore and PRR per quarter. Returns [{quarter, adverse_score, prr, report_count}]."""
        boundaries = self._compute_quarter_boundaries(num_quarters)
        pharm_class = None
        if target_symptom:
            pharm_class = self._discover_drug_class(drug_name)

        time_series = []
        for label, start, end in boundaries:
            raw = self.fetch_events(drug_name, patient_age=patient_age, patient_sex=patient_sex, start_date=start, end_date=end)
            reports = self._flatten_results(raw) if raw else []
            report_count = len(reports)

            if reports:
                result = self.calculate_final_score(
                    drug_name, reports, skip_benchmark=True,
                    patient_age=patient_age, patient_sex=patient_sex,
                    target_symptom=target_symptom
                )
                score = result["clinical_signal"]["adverse_score"]
            else:
                score = 0

            prr_value = None
            if target_symptom and pharm_class:
                drug_counts = self._fetch_symptom_counts(
                    drug_name=drug_name, patient_age=patient_age,
                    patient_sex=patient_sex, start_date=start, end_date=end
                )
                class_counts = self._fetch_symptom_counts(
                    pharm_class=pharm_class, patient_age=patient_age,
                    patient_sex=patient_sex, start_date=start, end_date=end
                )
                a = drug_counts.get(target_symptom.upper(), 0)
                a_plus_b = sum(drug_counts.values()) or 1
                c = class_counts.get(target_symptom.upper(), 0)
                c_plus_d = sum(class_counts.values()) or 1
                if a >= 3 and c > 0 and c_plus_d > 0 and a_plus_b > 0:
                    prr_value = round((a / a_plus_b) / (c / c_plus_d), 2)

            time_series.append({
                "quarter": label,
                "adverse_score": round(score),
                "prr": prr_value,
                "report_count": report_count,
            })
        return time_series

    def compute_trend(self, time_series):
        """Classify trend: RISING/STABLE/DECLINING/INSUFFICIENT_DATA."""
        valid = [q for q in time_series if q.get("report_count", 0) > 0]
        if len(valid) < 2:
            return "INSUFFICIENT_DATA"
        recent = valid[-1]["adverse_score"]
        comparison = valid[-3]["adverse_score"] if len(valid) >= 3 else valid[0]["adverse_score"]
        delta = recent - comparison
        if delta >= 10:
            return "RISING"
        elif delta <= -10:
            return "DECLINING"
        return "STABLE"

    def fetch_events(self, drug_name: str, patient_age: int = None, patient_sex: str = None, start_date: str = None, end_date: str = None): # type: ignore
        '''
        Executes the API call using the query builder and the session.
        Note on pagination: openFDA caps results at limit=1000 and skip+limit<=26000.
        We fetch 500 reports as a representative sample. The downstream confidence metric
        accounts for sample size, and the count endpoints used for PRR aggregate server-side
        with no pagination cap. Fetching all reports is not practical for real-time scoring.
        '''
        query_params = self.build_query(drug_name, patient_age=patient_age, patient_sex=patient_sex, start_date=start_date, end_date=end_date)
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
        #openFDA label endpoint
        label_url = 'https://api.fda.gov/drug/label.json'

        safe_name = self._sanitize_for_query(drug_name)
        query = f'search=openfda.brand_name:"{safe_name}"&limit=1'

        try:
            response = self.session.get(f"{label_url}?{query}&api_key={self.api_key}", timeout=10)
            response.raise_for_status()
            data = response.json()

            #FDA returns a list of strings for the adverse_reactions field
            results = data.get('results', [])
            if results:
                reactions_section = results[0].get('adverse_reactions', [])
                return " ".join(reactions_section).lower()
            return ""
        except Exception:
            #if label fails we assume 'unlabeled' and just return an empty string
            return ""

    def calculate_label_penalty(self, symptoms: str, label_text: str, is_serious: bool) -> float:
        '''
        Apply the penalty factors defined.
        Unlabeled + Serious: 2.0x | Unlabeled + Non-Serious: 1.5x | Labeled: 1.0x
        '''
        if not label_text:
            return 2.0 if is_serious else 1.5

        symptom_list = [s.strip().lower() for s in symptoms.split(",") if s.strip()]
        if not symptom_list:
            return 2.0 if is_serious else 1.5
        is_labeled = any(s in label_text for s in symptom_list)

        if not is_labeled:
            return 2.0 if is_serious else 1.5
        return 1.0

    def _calculate_report_score(self, report: dict, label_text: str) -> float:
        '''
        Calculates a weighted score for a single adverse event
        Logic: (Base Weight * Label Penalty) * Recency Factor
        '''
        #determine base severity weight
        is_serious = report.get('severity') == 'Serious'
        symptoms = report.get('symptoms', '')

        base_weight = self.SEVERITY_WEIGHTS['NON_SERIOUS']
        if report.get('is_death'):
            base_weight = self.SEVERITY_WEIGHTS['DEATH']
        elif is_serious and report.get('is_hospitalization'):
            base_weight = self.SEVERITY_WEIGHTS['HOSPITALIZATION']
        elif is_serious:
            base_weight = self.SEVERITY_WEIGHTS['OTHER_SERIOUS']

        #Apply label awareness penalty
        penalty = self.calculate_label_penalty(symptoms, label_text, is_serious)

        #calculate weighted signal
        raw_score = base_weight * penalty
        return raw_score

    def _classify_label_status(self, label_text: str, symptoms_str: str) -> str:
        '''
        Classifies a symptom string as LABELED, UNLABELED, or LABEL_STATUS_UNKNOWN
        based on whether any symptom appears in the drug's official FDA label text.
        '''
        if not label_text:
            return "LABEL_STATUS_UNKNOWN"
        symptom_list = [s.strip().lower() for s in symptoms_str.split(",") if s.strip()]
        if not symptom_list:
            return "LABEL_STATUS_UNKNOWN"
        if any(s in label_text for s in symptom_list):
            return "LABELED"
        return "UNLABELED"

    def _discover_drug_class(self, drug_name: str) -> str:
        '''
        Discovers the primary therapeutic class using frequency analysis on historical adverse event data, bypassing label noise
        '''
        url = "https://api.fda.gov/drug/event.json"
        #search by brand or generic name
        target_name = drug_name.upper()
        safe_name = self._sanitize_for_query(target_name)
        search_str = f'patient.drug.openfda.brand_name:"{safe_name}"'

        #safely encode the url to prevent 400 request errors
        encoded_search = urllib.parse.quote(search_str)
        count_param = "patient.drug.openfda.pharm_class_epc.exact"

        try:
            full_url = f"{url}?search={encoded_search}&count={count_param}&api_key={self.api_key}"
            print(f"[Discovery] Attempting algorithmic class mapping for {target_name}...")

            response = self.session.get(full_url, timeout=10)
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
        encoded_search = urllib.parse.quote(search_value)

        try:
            full_url = f"{url}?search={encoded_search}&limit=5&api_key={self.api_key}"
            response = self.session.get(full_url, timeout=10)
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            res = response.json()

            # Final Safety Net: Ban excipients from the fallback
            ignore_classes = ["Endoglycosidase [EPC]", "Hyaluronidase"]
            
            for result in res.get('results', []):
                epc_list = result.get('openfda', {}).get('pharm_class_epc', [])
                for epc in epc_list:
                    if epc not in ignore_classes:
                        return epc
            return ""
        except Exception:
            return ""
        
    def _discover_peers(self, pharm_class: str, target_drug: str) -> list: # type: ignore
        '''
        Find the top 3 most prescribed / reported peer drugs in the same pharmacologic class.
        Uses the openFDA event count endpoint
        '''
        if not pharm_class:
            return []
        
        print(f"[Discovery] Pharmacologic Class Identified: {pharm_class}")
        url = "https://api.fda.gov/drug/event.json"

        #clean the class name for the URL — must go through _sanitize_for_query per the security invariant
        clean_class = self._sanitize_for_query(pharm_class).replace(' ', '+')
        query = f'search=patient.drug.openfda.pharm_class_epc:"{clean_class}"&count=patient.drug.medicinalproduct.exact'

        try:
            response = self.session.get(f"{url}?{query}&api_key={self.api_key}", timeout=10)
            response.raise_for_status()
            data = response.json()

            peers = []
            target_upper = target_drug.upper()

            #FDA returns a list of terms and counts 
            for item in data.get('results', []):
                peer_name = item.get('term', '').upper()
                #exclude the target drug itself and invalid short names
                if peer_name and peer_name != target_upper and len(peer_name) > 3:
                    peers.append(peer_name)
                if len(peers) >= 3:
                    break
            
            print(f"[Discovery] Top Peers Found: {', '.join(peers)}")
            return peers
        
        except Exception as e:
            print(f"[Discovery] Failed to find peers for class {pharm_class}.")
            return []

    
    def get_peer_benchmark(self, drug_name: str, patient_age: int = None, patient_sex: str = None) -> float: #type: ignore case
        '''
        Calculates the average AdverseScore of peer drugs using Dynamic Ontology Mapping.
        '''
        target_upper = drug_name.upper()

        #trigger the dynamic discovery
        pharm_class = self._discover_drug_class(target_upper)
        peers = self._discover_peers(pharm_class, target_upper)

        if not peers:
            print(f'[Benchmarking] No peers discovered for {drug_name}. Skipping benchmark.')
            return 0.0
        
        peer_scores = []
        print(f"[Benchmarking] Evaluating {drug_name} against the discovered peers: {', '.join(peers)}")

        for peer in peers:
            print(f'[Benchmarking] Fetching clinical data for {peer}...')
            raw = self.fetch_events(peer, patient_age, patient_sex)
            clean = self._flatten_results(raw)
            if not clean:
                print(f'[Benchmarking] No data for peer {peer}, excluding from benchmark.')
                continue
            result = self.calculate_final_score(peer, clean, skip_benchmark=True)
            peer_score = result['clinical_signal']['adverse_score']
            peer_scores.append(peer_score)
        
        return round(sum(peer_scores) / len(peer_scores), 2) if peer_scores else 0.0

    def _calculate_confidence(self, clean_reports: list) -> dict:
        '''
        Evaluates the statistical reliability of the AdverseScore. 
        Analyzes sample size (N) and data completeness
        '''
        total_reports = len(clean_reports)
        if total_reports == 0:
            return {'level': 'None', 'metric': 0.0, 'defect_ratio': 0.0}
        
        # METHODOLOGY NOTE: This is a step function with sharp discontinuities:
        # 49 reports → 40.0, 50 reports → 90.0 (a 50-point jump for +1 report).
        # A continuous function like min(100, N * 1.25) or a log curve would avoid
        # this cliff effect. The current thresholds are empirically chosen.
        if total_reports >= 80:
            base_confidence = 100.0
        elif total_reports >= 50:
            base_confidence = 90.0
        else:
            base_confidence = 40.0
        
        #Quality Assessment
        low_quality_counts = sum(
            1 for r in clean_reports
            if not r.get('date') or r.get('symptoms') == 'Unknown'
        )
        quality_defect_ratio = low_quality_counts / total_reports


        #Elegant Penalty
        final_confidence_score = base_confidence - (quality_defect_ratio * 50)
        final_confidence_score = max(0.0, final_confidence_score)

        #Categorical Level for the AI Prompt
        if final_confidence_score >= 80:
            level = 'High'
        elif final_confidence_score >= 50:
            level = 'Medium'
        else:
            level = 'Low'
        
        return {
            'level': level,
            'metric': round(final_confidence_score, 1),
            'defect_ratio': round(quality_defect_ratio, 2),
        }

    def _generate_guardrails(self, adverse_score: float, confidence_metrics: dict, prr_metrics: Optional[dict] = None) -> dict:
        '''
        Generates deterministic boolean flags to control AI behavior
        Prevents hallucination and ensures clinical safety protocols
        '''
        confidence_level = confidence_metrics.get('level', 'Low')

        diagnosis_lock = True

        #human in the loop triggers
        requires_human_review = False
        if adverse_score > 70:
            requires_human_review = True
        elif adverse_score > 40 and confidence_level == 'Low':
            requires_human_review = True
        
        #specialist routing
        route_to_specialist = adverse_score > 60

        #PRR override
        if prr_metrics and prr_metrics.get("signal_detected"):
            requires_human_review = True
            route_to_specialist = True

        return {
            "diagnosis_lock": diagnosis_lock,
            "requires_human_review": requires_human_review,
            "route_to_specialist": route_to_specialist,
            "system_directive": "Halt autonomous clinical advice if requires_human_review is True."
        }

    def _fetch_symptom_counts(self, drug_name: str = None, pharm_class: str = None, patient_age: int = None, patient_sex: str = None, start_date: str = None, end_date: str = None) -> dict: #type: ignore case
        '''
        Hits the openFDA count endpoint for a specific drug or a pharmacologic class - forces the FDA servers to aggregate symptom frequencies instantly
        '''
        query_parts = []

        if drug_name:
            # applied to build_query and fetch_label_text).
            safe_name = self._sanitize_for_query(drug_name)
            query_parts.append(f'patient.drug.medicinalproduct:"{safe_name}"')
        elif pharm_class:
            clean_class = pharm_class.replace(' ', '+').replace('"', '')
            query_parts.append(f'patient.drug.openfda.pharm_class_epc:"{clean_class}"')

        if patient_sex:
            # 1=Male, 2=Female. The old code mapped F→1 (Male) and M→2 (Female).
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
            # throws KeyError if openFDA returns a result item missing either field. This
            # crashes the entire dict comprehension. Using .get() with a skip guard handles
            # malformed items gracefully instead of aborting the whole aggregation.
            counts = {}
            for item in data.get('results', []):
                term = item.get('term')
                count = item.get('count')
                if term is not None and count is not None:
                    counts[term.upper()] = count
            return counts
        except Exception:
            return {}
    
    def _calculate_prr_metrics(self, drug_name: str, pharm_class: str, target_symptom: str, patient_age: int = None, patient_sex: str = None, label_text: str = "") -> dict: #type: ignore case
        '''
        Executes the PRR ratio calculation and 95% confidence interval math.

        METHODOLOGY NOTE (pharmacovigilance deviation from Evans et al., 2001):
        1. The classical PRR comparator is "all OTHER drugs" (excluding the target).
           This implementation uses "all drugs in the same pharmacologic class" which
           INCLUDES the target drug in the denominator. When the class has many drugs
           the impact is small, but for small classes this dilutes the signal.
        2. The denominators (a+b, c+d) are total symptom MENTION counts from the
           openFDA count endpoint, not total REPORT counts. Since a single report can
           list multiple symptoms, this inflates the denominators and slightly deflates
           the PRR vs. the classical report-level formulation.
        Both are accepted simplifications in automated signal detection systems where
        individual report-level 2x2 tables are impractical to construct from the API.
        '''
        drug_counts = self._fetch_symptom_counts(drug_name=drug_name, patient_age=patient_age, patient_sex=patient_sex)
        class_counts = self._fetch_symptom_counts(pharm_class=pharm_class, patient_age=patient_age, patient_sex=patient_sex)

        symptom_upper = target_symptom.upper()

        #a = target_drug + target_symptom 
        #a_plus_b = Total Target Drug Symptoms
        a = drug_counts.get(symptom_upper, 0)
        a_plus_b = sum(drug_counts.values())

        #c = class + target symptom
        # #c_plus_d = total class symptoms
        c = class_counts.get(symptom_upper, 0)
        c_plus_d = sum(class_counts.values())

        #classify label status for the target symptom
        symptom_label_status = self._classify_label_status(label_text, target_symptom)

        #guard against division by zero or stat insignificant sample size
        if a < 3 or c == 0 or a_plus_b == 0 or c_plus_d == 0:
            return {"prr": 0.0, "ci_lower": 0.0, "signal_detected": False, "target_symptom": symptom_upper, "drug_cases": a, "class_cases": c, "label_status": symptom_label_status}

        prr = (a / a_plus_b) / (c / c_plus_d)

        try:
            # 95% CI lower bound for log-transformed PRR (Wald method).
            # SE(ln(PRR)) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
            # The radicand is always >= 0 because each pair (1/x - 1/(x+y)) = y/(x*(x+y)) >= 0
            # given the guards above ensure a,c > 0 and a<=a+b, c<=c+d. The ValueError
            # catch below is a defensive fallback that should never trigger in practice.
            se = math.sqrt((1/a) + (1/c) - (1/a_plus_b) - (1/c_plus_d))
            ci_lower = math.exp(math.log(prr) - 1.96 * se)
        except ValueError:
            ci_lower = 0.0

        #Mathematic Guardrail
        signal_detected = ci_lower > 1.0 and a >= 3

        if signal_detected:
            print(f"[Math Engine] Stat Signal Detected: {symptom_upper} | PRR: {round(prr,2)} | CI Lower: {round(ci_lower, 2)}")

        return {
            "prr": round(prr, 2),
            "ci_lower": round(ci_lower, 2),
            "signal_detected": signal_detected,
            "target_symptom": target_symptom,
            "drug_cases": a,
            "class_cases": c,
            "label_status": symptom_label_status
        }    

    
    def calculate_final_score(self, drug_name: str, clean_reports: list, skip_benchmark: bool = False, patient_age: int = None, patient_sex: str = None, target_symptom: str = None) -> dict: #type: ignore case
        '''
        Aggregates individual report scores into a final Adverse Score for the drug
        Implements a Recency decay and Normalization logic
        '''
        if not clean_reports:
            return {
                "metadata": {
                    "tool_name": "AdverseScore",
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "clinical_disclaimer": "This score is for informational purposes only and does not constitute medical advice. The responsibility ultimately remains with the clinician."
                },
                "clinical_signal": {
                    "drug_target": drug_name.upper(),
                    "adverse_score": 0.0,
                    "status": "Incomplete Data",
                    "relative_risk": "N/A",
                    "class_benchmark_avg": 0.0
                },
                "data_integrity": {
                    "report_count": 0,
                    "confidence_level": "None",
                    "defect_ratio": 0.0
                },
                "agent_directives": {
                    "diagnosis_lock": True,
                    "requires_human_review": False,
                    "route_to_specialist": False,
                    "system_directive": "Inform the user that insufficient safety data exists in the openFDA database for this drug. Suggest they verify the drug name spelling and try the exact brand or generic name. Do not attempt to calculate a risk profile."
                }
            }
        
        #fetch label text once for this drug to use in the loop
        label_text = self.fetch_label_text(drug_name)

        #determine overall label status from all report symptoms
        all_symptoms = ", ".join(r.get("symptoms", "") for r in clean_reports)
        label_status = self._classify_label_status(label_text, all_symptoms)

        prr_metrics = None
        if target_symptom:
            pharm_class = self._discover_drug_class(drug_name)
            if pharm_class:
                prr_metrics = self._calculate_prr_metrics(drug_name, pharm_class, target_symptom, patient_age, patient_sex, label_text=label_text)

        total_weighted_points = 0
        ninety_days_ago = datetime.now() - timedelta(days=90)

        for report in clean_reports:
            report_score = self._calculate_report_score(report, label_text)

            try:
                report_date = datetime.strptime(report['date'], "%Y%m%d")
                decay_multiplier = 1.0 if report_date >= ninety_days_ago else 0.5
            except (ValueError, TypeError):
                # METHODOLOGY NOTE: Reports with missing or unparseable dates default to
                # full recency weight (1.0). This means low-quality reports (which the
                # confidence metric separately penalizes) are treated as maximally recent,
                # slightly inflating the score. An alternative would be 0.5 (assume old)
                # or 0.75 (neutral). The current choice errs toward surfacing signals.
                decay_multiplier = 1.0
            
            total_weighted_points += (report_score * decay_multiplier)
        
        mean_signal = total_weighted_points / len(clean_reports)
        # METHODOLOGY NOTE: The x40 scalar maps the mean weighted signal into a 0-100
        # range. The theoretical max mean_signal is 1.75 * 2.0 * 1.0 = 3.5 (all death
        # + unlabeled + recent), yielding 3.5 * 40 = 140, capped to 100. The theoretical
        # min for a non-empty dataset is 0.25 * 1.0 * 0.5 = 0.125 (all non-serious +
        # labeled + old), yielding 0.125 * 40 = 5.0. This scalar is empirically chosen
        # and not derived from a statistical model.
        final_score = min(100, round(mean_signal * 40, 2))

        #Add f-string formatting
        status = 'Stable'
        if final_score > 70: status = 'High Signal - Urgent Review'
        elif final_score > 30: status = f'Monitor - Emerging Trend for {drug_name}'

        #execute benchmarking 
        benchmark_avg = 0.0
        relative_risk = 'N/A'
        if not skip_benchmark:
            benchmark_avg = self.get_peer_benchmark(drug_name, patient_age, patient_sex)
            if benchmark_avg > 0:
                relative_risk = 'Average'
                if final_score > (benchmark_avg * 1.5):
                    relative_risk = 'Elevated vs Class Peers'
                elif final_score < (benchmark_avg * 0.7):
                    relative_risk = 'Lower than Class Peers'
        
        #integrate calculate confidence
        confidence_metrics = self._calculate_confidence(clean_reports)

        #integrate guardrails
        guardrails = self._generate_guardrails(final_score, confidence_metrics, prr_metrics)

        #integrate the payload schema
        agent_payload = {
            "metadata": {
                "tool_name": "AdverseScore",
                "version": "1.0",
                "timestamp": datetime.now().isoformat() + "Z",
                "clinical_disclaimer": "This score is for informational purposes only and does not constitute medical advice. The responsibility ultimately remains with the clinician."
            },
            "clinical_signal": {
                "drug_target": drug_name.upper(),
                "adverse_score": final_score,
                "status": status,
                "relative_risk": relative_risk,
                "class_benchmark_avg": benchmark_avg,
                "label_status": label_status
            },
            "data_integrity": {
                "report_count": len(clean_reports),
                "confidence_level": confidence_metrics.get("level"),
                "defect_ratio": confidence_metrics.get("defect_ratio")
            },
            "pharmacovigilance_metrics": prr_metrics,
            "agent_directives": guardrails
        }

        return agent_payload


#Quick Exection
if __name__ == "__main__":
    client = AdverseScoreClient()
    #test with a common drug
    drug_name = "KEYTRUDA"

    #Creating variables with default states
    raw_data = None
    clean_list = []

    print(f'Querying openFDA for {drug_name} signals')
    raw_data = client.fetch_events(drug_name)

    if raw_data:
        #flatten the data
        clean_list = client._flatten_results(raw_data)
        print(f'Successfully processed {len(clean_list)} reports for {drug_name}.')

    final_result = client.calculate_final_score(drug_name, clean_list)
    print(json.dumps(final_result, indent=2))
    
    print("\n--- ADVERSE SCORE SUMMARY ---")
    print(f"Drug:          {final_result['clinical_signal']['drug_target']}")
    print(f"Score:         {final_result['clinical_signal']['adverse_score']}/100")
    print(f"Status:        {final_result['clinical_signal']['status']}")
    print(f"Reports:       {final_result['data_integrity']['report_count']}")
    print(f"Peer Average:  {final_result['clinical_signal']['class_benchmark_avg']}")
    print(f"Relative Risk: {final_result['clinical_signal']['relative_risk']}")
    print(f"Confidence:    {final_result['data_integrity']['confidence_level']}")
    print(f"Diagnosis Lock:{final_result['agent_directives']['diagnosis_lock']}")
    print("------------------------------")