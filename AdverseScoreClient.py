import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from config import initialize_config



class AdverseScoreClient:
    base_url = "https://api.fda.gov/drug/event.json"

    #Class Attributes 
    SEVERITY_WEIGHTS = {
        'DEATH': 1.75,
        'HOSPITALIZATION': 1.0,
        'OTHER_SERIOUS': 0.75,
        'NON_SERIOUS': 0.25
    }

    PEER_GROUPS = {
        'KEYTRUDA': ['OPDIVO', 'TECENTRIQ', 'BAVENCIO'],
        'PEMBROLIZUMAB': ['NIVOLUMAB', 'ATEZOLIZUMAB', 'AVELUMAB']
    }

    def __init__(self):
        self.api_key: str = initialize_config()
        self.session = self._get_transport_session()


    def build_query(self, drug_name: str, days_back: int = 365) -> str:
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

        #clean and encode 
        encoded_search = search_params.replace(" ", "+")

        return f"search={encoded_search}&limit=500"

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

    def fetch_events(self, drug_name: str):
        '''
        Executes the API call using the query builder and the session
        '''
        query_params = self.build_query(drug_name)
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
            reactions = [r.get('reactionmeddrapt', 'Unknown') for r in report.get('patient', {}).get('reaction', [])]

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
    
    def get_peer_benchmark(self, drug_name: str) -> float:
        '''
        Calculates the average AdverseScore of peer drugs
        '''
        target_upper = drug_name.upper()
        peers = self.PEER_GROUPS.get(target_upper, [])

        if not peers:
            return 0.0
        
        peer_scores = []
        print(f"Benchmarking {drug_name} against peers: {', '.join(peers)}")

        for peer in peers:
            raw = self.fetch_events(peer)
            clean = self._flatten_results(raw)
            result = self.calculate_final_score(peer, clean, skip_benchmark=True)
            peer_scores.append(result['adverse_score'])
        
        return round(sum(peer_scores) / len(peer_scores), 2) if peer_scores else 0.0

    def _calculate_confidence(self, clean_reports: list) -> dict:
        '''
        Evaluates the statistical reliability of the AdverseScore. 
        Analyzes sample size (N) and data completeness
        '''
        total_reports = len(clean_reports)
        if total_reports == 0:
            return {'level': 'None', 'score': 0.0, 'reason': 'No data available'}
        
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

    
    def calculate_final_score(self, drug_name: str, clean_reports: list, skip_benchmark: bool = False) -> dict:
        '''
        Aggregates individual report scores into a final Adverse Score for the drug
        Implements a Recency decay and Normalization logic
        '''
        if not clean_reports:
            return {
            'drug': drug_name,
            'adverse_score': 0,
            'status': 'Incomplete Data',
            'report_count': 0,
            'benchmark_avg': 0.0,
            'relative_risk': 'N/A'
            }
        
        #fetch label text once for this drug to use in the loop 
        label_text = self.fetch_label_text(drug_name)

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
            benchmark_avg = self.get_peer_benchmark(drug_name)
            if benchmark_avg > 0:
                relative_risk = 'Average'
                if final_score > (benchmark_avg * 1.5):
                    relative_risk = 'Elevated vs Class Peers'
                elif final_score < (benchmark_avg * 0.7):
                    relative_risk = 'Lower than Class Peers'
        
        #integrate calculate confidence
        confidence_metrics = self._calculate_confidence(clean_reports)

        return {
            'drug': drug_name,
            'adverse_score': final_score,
            'status': status,
            'report_count': len(clean_reports),
            'benchmark_avg': benchmark_avg,
            'relative_risk': relative_risk,
            'confidence': confidence_metrics
        }


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
    
    print("\n--- ADVERSE SCORE SUMMARY ---")
    print(f"Drug:          {final_result['drug']}")
    print(f"Score:         {final_result['adverse_score']}/100")
    print(f"Status:        {final_result['status']}")
    print(f"Reports:       {final_result['report_count']}")
    print(f"Peer Average:  {final_result['benchmark_avg']}")
    print(f"Relative Risk: {final_result['relative_risk']}")
    print("------------------------------")