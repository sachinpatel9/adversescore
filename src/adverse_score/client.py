import math
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from .config import initialize_config
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


    def build_query(self, drug_name: str, days_back: int = 365, limit: int = 500, patient_age: int = None, patient_sex: str = None) -> str:  # type: ignore
        '''
        Constructs a valid openFDA Lucene search query.
        Example output: search=patient.drug.medicinalproduct:'TYLENOL'+AND+receivedate:[20231210+TO+20240310]
        '''

        #Handle dates: YYYYMMDD format
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
        date_range = f"[{start_date}+TO+{end_date}]"

        #building the search parameters 
        search_params = f'patient.drug.medicinalproduct:"{drug_name}" AND receivedate:{date_range}'

        #Inject Demographic Filters
        if patient_sex:
            #openFDA sex coding 
            sex_code = "1" if patient_sex.upper() == "F" else "2"
            search_params += f' AND patient.patientsex:{sex_code}'

        if patient_age:
            #create a 10 year age cohort bracket to ensure adequate sample size
            lower_bound = max(0, patient_age - 5)
            upper_bound = patient_age + 5
            search_params += f' AND patient.patientonsetage:[{lower_bound}+TO+{upper_bound}]'
        
        encoded_search = search_params.replace(" ", "+")

        return f"search={encoded_search}&limit={limit}"


        #clean and encode 
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

    def fetch_events(self, drug_name: str, patient_age: int = None, patient_sex: str = None): # type: ignore
        '''
        Executes the API call using the query builder and the session
        '''
        query_params = self.build_query(drug_name, patient_age=patient_age, patient_sex=patient_sex)
        full_url = f"{self.base_url}?{query_params}&api_key={self.api_key}"

        try:
            response = self.session.get(full_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Integration Error: Failed to fetch data for {drug_name}. Error: {e}")
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
        query = f'search=openfda.brand_name:"{drug_name}"&limit=1'

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
        
        #check if any reported symptoms is MISSING from the label text - implementing simple key word matching for now, could be improved with NLP techniques
        symptom_list = [s.strip().lower() for s in symptoms.split(",")]
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

        #check specific FDA flags for the higher 'Death' weight
        base_weight = self.SEVERITY_WEIGHTS['NON_SERIOUS']
        if report.get('is_death'):
            base_weight = self.SEVERITY_WEIGHTS['DEATH']
        elif is_serious:
            base_weight = self.SEVERITY_WEIGHTS['HOSPITALIZATION']

        #Apply label awareness penalty
        penalty = self.calculate_label_penalty(symptoms, label_text, is_serious)

        #calculate weighted signal
        raw_score = base_weight * penalty
        return raw_score
    
    def _discover_drug_class(self, drug_name: str) -> str:
        '''
        Queries the FDA label database to find the Established Pharmacologic Class (EPC)
        '''
        url = "https://api.fda.gov/drug/label.json"
        #search by brand or generic name
        query = f'search=openfda.brand_name:"{drug_name}"+OR+openfda.generic_name:"{drug_name}"&limit=1'

        try:
            response = self.session.get(f"{url}?{query}&api_key={self.api_key}",
                                        timeout=10)
            response.raise_for_status()
            data = response.json()

            #extract pharm_class_epc
            results = data.get('results', [])
            if results:
                openfda_data = results[0].get('openfda', {})
                epc_list = openfda_data.get('pharm_class_epc', [])
                if epc_list:
                    return epc_list[0] #return the primary class
            return ""
        except Exception as e:
            print(f"[Discovery] Failed to identify pharmacology class for {drug_name}.")
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

        #clean the class name for the URL
        clean_class = pharm_class.replace(' ', '+').replace('"', '')
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
        
        #Volume Assessment
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

    def _generate_guardrails(self, adverse_score: float, confidence_metrics: dict, prr_metrics: dict = None) -> dict:
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

    def _fetch_symptom_counts(self, drug_name: str = None, pharm_class: str = None, patient_age: int = None, patient_sex: str = None) -> dict: #type: ignore case
        '''
        Hits the openFDA count endpoint for a specific drug or a pharmacologic class - forces the FDA servers to aggregate symptom frequencies instantly
        '''
        query_parts = []

        if drug_name:
            query_parts.append(f'patient.drug.medicinalproduct:"{drug_name}"')
        elif pharm_class:
            clean_class = pharm_class.replace(' ', '+').replace('"', '')
            query_parts.append(f'patient.drug.openfda.pharm_class_epc:"{clean_class}"')
        
        if patient_sex:
            sex_code = "1" if patient_sex.upper() == "F" else "2"
            query_parts.append(f'patient.patientsex:{sex_code}')
        if patient_age:
            lower = max(0, patient_age - 5)
            upper = patient_age + 5
            query_parts.append(f'patient.patientonsetage:[{lower}+TO+{upper}]')
        
        search_string = "+AND+".join([q.replace(" ", "+") for q in query_parts])
        url = f"{self.base_url}?search={search_string}&count=patient.reaction.reactionmeddrapt.exact&limit=1000&api_key={self.api_key}"

        try:
            entity = drug_name or pharm_class
            print(f"[Aggreagation] Running server-side symptom count for {entity}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            #map symptom named directly to their integer counts
            return {item['term'].upper(): item['count'] for item in data.get('results', [])}
        except Exception:
            return {}
    
    def _calculate_prr_metrics(self, drug_name: str, pharm_class: str, target_symptom: str, patient_age: int = None, patient_sex: str = None) -> dict: #type: ignore case
        '''
        Executes the PRR ratio calculation and 95% confidence interval math
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

        #guard against division by zero or stat insignificant sample size
        if a < 3 or c == 0 or a_plus_b == 0 or c_plus_d == 0:
            return {"prr": 0.0, "ci_lower": 0.0, "signal_detected": False, "target_symptom": symptom_upper, "drug_cases": a, "class_cases": c}

        prr = (a / a_plus_b) / (c / c_plus_d)

        try:
            #95% CI lower bound for PRR
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
            "class_cases": c
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
                    "system_directive": "Inform the user that insufficient safety data exists in the openFDA database for this drug. Do not attempt to calculate a risk profile."
                }
            }
        
        #fetch label text once for this drug to use in the loop 
        label_text = self.fetch_label_text(drug_name)

        prr_metrics = None
        if target_symptom:
            pharm_class = self._discover_drug_class(drug_name)
            if pharm_class:
                prr_metrics = self._calculate_prr_metrics(drug_name, pharm_class, target_symptom, patient_age, patient_sex)

        total_weighted_points = 0
        ninety_days_ago = datetime.now() - timedelta(days=90)

        for report in clean_reports:
            report_score = self._calculate_report_score(report, label_text)

            try:
                report_date = datetime.strptime(report['date'], "%Y%m%d")
                decay_multiplier = 1.0 if report_date >= ninety_days_ago else 0.5
            except (ValueError, TypeError):
                decay_multiplier = 1.0
            
            total_weighted_points += (report_score * decay_multiplier)
        
        mean_signal = total_weighted_points / len(clean_reports)
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
                "class_benchmark_avg": benchmark_avg
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