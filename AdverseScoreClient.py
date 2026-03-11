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

        return f"search={encoded_search}&limit=100"

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

        #reuse self.session defined at the class level
        with self.session as session:

            try:
                response = session.get(full_url, timeout=10)
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
        with self.session as session:
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
        if is_serious:
            base_weight = self.SEVERITY_WEIGHTS['HOSPITALIZATION']

        #Apply label awareness penalty
        penalty = self.calculate_label_penalty(symptoms, label_text, is_serious)

        #calculate weighted signal
        raw_score = base_weight * penalty

        return raw_score
    
    def calculate_final_score(self, drug_name: str, clean_reports: list) -> dict:
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
            }
        
        #fetch label text once for this drug to use in the loop 
        label_text = self.fetch_label_text(drug_name)

        total_weighted_points = 0
        ninety_days_ago = datetime.now() - timedelta(days=90)

        for report in clean_reports:
            #calculate base weighted score
            if report.get('is_death'):
                base_weight = self.SEVERITY_WEIGHTS['DEATH']
            elif report.get('severity') == 'Serious':
                base_weight = self.SEVERITY_WEIGHTS['HOSPITALIZATION']
            else:
                base_weight = self.SEVERITY_WEIGHTS['NON_SERIOUS']
            
            #apply label penalty
            penalty = self.calculate_label_penalty(report['symptoms'], label_text, report['severity'] == 'Serious')
            report_score = base_weight * penalty

            #apply recency decay (3 month delay)
            report_date = datetime.strptime(report['date'], "%Y%m%d")
            decay_multiplier = 1.0 if report_date >= ninety_days_ago else 0.5
            
            total_weighted_points += (report_score * decay_multiplier)
        
        #Normalize & Scale
        mean_signal = total_weighted_points / len(clean_reports)

        #scaling factor: adjusted to make 'serious' / 'unlabeled' events spike the score
        final_score = min(100, round(mean_signal * 40, 2))

        #determine status tone
        status = 'Stable'
        if final_score > 70: status = 'High Signal - Urgent Review'
        elif final_score > 30: status = f'Monitor - Emerging Trend for {drug_name}'
        

        return {
            'drug': drug_name,
            'adverse_score': final_score,
            'status': status,
            'report_count': len(clean_reports),
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
    print(f"Drug:    {final_result['drug']}")
    print(f"Score:   {final_result['adverse_score']}/100")
    print(f"Status:  {final_result['status']}")
    print(f"Reports: {final_result['report_count']}")
    print("------------------------------")