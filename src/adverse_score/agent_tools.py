import json
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from .client import AdverseScoreClient

#instantiate the client globally so the HTTP session persists across multiple agent calls
_global_client = AdverseScoreClient()

#Strict Extraction Schema (Pydantic)
class ClinicalQuerySchema(BaseModel):
    drug_name: str = Field(
        ...,
        description="The exact brand or generic name of the target drug (ex. KEYTRUDA, OZEMPIC)."
    )
    patient_age: Optional[int] = Field(
        None,
        description='The exact age of the patient in years, if provided in the prompt.'
    )
    patient_sex: Optional[str] = Field(
        None,
        description="The biological sex of the patient, strictly 'M' or 'F', if provided in the prompt."
    )

#Agent Tool 
@tool(args_schema=ClinicalQuerySchema)
def get_adverse_score(drug_name: str, patient_age: int= None, patient_sex: str= None) -> str: # type: ignore
    '''
    Calculates a clinial safety risk score (0-100) based on FDA adverse event reports,
    Call this tool when a user asks about the safety, side effects, toxicity, or risk profile of a specific medication.
    Extracts demographics if provided to establish clinical context.
    '''
    print(f"Tool Initiated: Fetching AdverseScore for {drug_name.upper()}")

    #Visual confirmation 
    if patient_age or patient_sex:
        print(f'[Agent Network] Demographics Extracted -> Age: {patient_age}, Sex: {patient_sex}')
    
    try:
        raw_data = _global_client.fetch_events(drug_name, patient_age, patient_sex)
        clean_list = _global_client._flatten_results(raw_data) if raw_data else []
        agent_payload = _global_client.calculate_final_score(
            drug_name, 
            clean_list,
            patient_age=patient_age,
            patient_sex=patient_sex)

        #Inject the parsed demographics back into the metadata so the UI can see them 
        agent_payload['metadata']['extracted_demographics'] = {
            'age': patient_age,
            'sex': patient_sex
        }

        return json.dumps(agent_payload)

    except Exception as e:
        error_payload = {
            'metadata': {'tool_name': 'AdverseScore', 'status': 'System Error'},
            'agent_directives': {
                'system_directive': f'Inform the user that the AdverseScore tool is currently unavailable. Internal Error: {str(e)}'
            }
        }
        return json.dumps(error_payload)
    
