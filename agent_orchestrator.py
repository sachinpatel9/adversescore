import json 
from langchain_core.tools import tool
from AdverseScoreClient import AdverseScoreClient

#instantiate the client globally so the HTTP session persists across multiple agent calls
_global_client = AdverseScoreClient()

#LangChain Tool Wrapper
@tool
def get_adverse_score(drug_name: str) -> str:
    '''
    calculates a clinical safety risk score (0-100) and extracts relative reporting rates
    for a specified pharmaceutical drug based on the FDA adverse event reports
    Call this tool when a user asks about the safety, side effects, toxicity, or risk profile of a specific medication.
    '''
    print(f'Tool Initiated: Fetching AdverseScore for {drug_name.upper()}')

    try:
        raw_data = _global_client.fetch_events(drug_name)
        clean_list = _global_client._flatten_results(raw_data) if raw_data else []
        agent_payload = _global_client.calculate_final_score(drug_name, clean_list)

        return json.dumps(agent_payload)
    
    except Exception as e:
        # Graceful degradation for the Agent
        error_payload = {
            "metadata": {"tool_name": "AdverseScore", "status": "System Error"},
            "agent_directives": {
                "system_directive": f"Inform the user that the AdverseScore tool is currently unavailable. Internal Error: {str(e)}"
            }
        }
        return json.dumps(error_payload)
    
# Quick verification test
if __name__ == "__main__":
    print(f"Tool Name: {get_adverse_score.name}")
    print(f"Tool Description: {get_adverse_score.description}") 