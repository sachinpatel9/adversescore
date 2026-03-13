import json 
from .client import AdverseScoreClient

adverse_score_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_adverse_score",
        "description": "Calculates a clinical safety risk score (0-100) and extracts relative reporting rates for a specified pharmaceutical drug based on FDA adverse event reports. Call this tool when a user asks about the safety, side effects, toxicity, or risk profile of a specific medication.",
        "parameters": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "The exact brand or generic name of the drug (e.g., 'KEYTRUDA', 'ATORVASTATIN')."
                }
            },
            "required": ["drug_name"]
        }
    }
}

def execute_adverse_score(drug_name: str) -> str:
    '''
    Function that the AI Agent will call. It returns a serialized JSON string for the LLM to parse
    '''

    try:
        #Instantiate the black box
        client = AdverseScoreClient()

        #run the pipeline
        raw_data = client.fetch_events(drug_name)

        #handle the empty state clearly
        clean_list = client._flatten_results(raw_data) if raw_data else []

        #generate the agent payload
        agent_payload = client.calculate_final_score(drug_name, clean_list)

        #return as a serialized string
        return json.dumps(agent_payload)
    
    except Exception as e:
        error_payload = {
            "metadata": {
                "tool_name": "AdverseScore",
                "status": "System Error"
            },
            "agent_directives": {
                "system_directive": f"Inform the user that the AdverseScore tool is currently unavailable due to an external system error. Do not attempt to guess the safety profile. Internal Error: {str(e)}"
            }
        }
        return json.dumps(error_payload)