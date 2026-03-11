import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta

#Integration: import the validator from config.py
from config import initialize_config

class AdverseScoreClient:
    base_url = "https://api.fda.gov/drug/event.json"

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

        session = self._get_transport_session()

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
                'severity': 'Serious' if report.get('seriousness') == 1 else 'Non-Serious',
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



#Quick Exection
if __name__ == "__main__":
    client = AdverseScoreClient()
    #test with a common drug
    drug_name = "TYLENOL"
    raw_data = client.fetch_events(drug_name)

    if raw_data:
        #flatten the data
        clean_list = client._flatten_results(raw_data)
        print(f'Successfully processed {len(clean_list)} reports for {drug_name}.')

        #display the first 5 reports
        for i, report in enumerate(clean_list[:5]):
            print(f"Report {i+1}: ID={report['report_id']}, Date={report['date']}, Severity={report['severity']}, Company={report['company']}")
    else:
        print("No data retrieved.")