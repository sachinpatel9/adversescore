import json
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from langchain_core.tools import tool
from .client import AdverseScoreClient
from .logger import get_logger, log_event

logger = get_logger("agent")

#instantiate the client globally so the HTTP session persists across multiple agent calls
_global_client = AdverseScoreClient()

#Strict Extraction Schema (Pydantic)
class ClinicalQuerySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_name: str = Field(
        ...,
        min_length=1,
        description="The exact brand or generic name of the target drug (ex. KEYTRUDA, OZEMPIC)."
    )
    patient_age: Optional[int] = Field(
        None,
        ge=1,
        le=120,
        description='The exact age of the patient in years (1-120), if provided in the prompt.'
    )
    patient_sex: Optional[Literal["M", "F"]] = Field(
        None,
        description="The biological sex of the patient, strictly 'M' or 'F', if provided in the prompt."
    )
    target_symptom: Optional[str] = Field(
        None,
        min_length=1,
        description="The specific adverse event, side effect, or symptom to analyze (eg. 'pancreatitis', 'fatigue') if provided."
    )
    include_temporal: Optional[bool] = Field(
        None,
        description="Set to true when the user asks about trends, changes over time, quarterly data, or historical safety patterns."
    )

    @field_validator("drug_name")
    @classmethod
    def normalize_drug_name(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("drug_name must not be blank after stripping whitespace")
        return stripped

#Agent Tool 
@tool(args_schema=ClinicalQuerySchema)
def get_adverse_score(drug_name: str, patient_age: int= None, patient_sex: str= None, target_symptom: str = None, include_temporal: bool = None) -> str: #type: ignore case
    '''
    Calculates a clinial safety risk score (0-100) based on FDA adverse event reports,
    Call this tool when a user asks about the safety, side effects, toxicity, or risk profile of a specific medication.
    Extracts demographics and specific target symptoms to execute Proportional Reporting Ration (PRR) analysis if requested.
    '''
    log_event(logger, "tool_invoked", drug=drug_name.upper())

    if patient_age or patient_sex:
        log_event(logger, "demographics_extracted", age=patient_age, sex=patient_sex)
    if target_symptom:
        log_event(logger, "symptom_extracted", symptom=target_symptom.upper())
    
    try:
        raw_data = _global_client.fetch_events(drug_name, patient_age, patient_sex)
        clean_list = _global_client._flatten_results(raw_data) if raw_data else []
        agent_payload = _global_client.calculate_final_score(
            drug_name, 
            clean_list,
            patient_age=patient_age,
            patient_sex=patient_sex,
            target_symptom=target_symptom # type: ignore
        )

        #Inject the parsed demographics back into the metadata so the UI can see them 
        agent_payload['metadata']['extracted_demographics'] = {
            'age': patient_age,
            'sex': patient_sex,
            'target_symptom': target_symptom
        }

        if include_temporal:
            log_event(logger, "temporal_requested", drug=drug_name.upper())
            time_series = _global_client.fetch_quarterly_data(
                drug_name, num_quarters=4,
                patient_age=patient_age, patient_sex=patient_sex,
                target_symptom=target_symptom
            )
            trend_classification = _global_client.compute_trend(time_series)
            agent_payload['temporal_analysis'] = {
                'time_series': time_series,
                'trend_classification': trend_classification,
            }

        # --- Persistence: save analysis and detect delta vs prior ---
        try:
            from .persistence import AnalysisStore
            store = AnalysisStore()
            prior = store.get_prior_analysis(drug_name)
            store.save_analysis(agent_payload)
            if prior:
                score_delta = agent_payload["clinical_signal"]["adverse_score"] - prior["adverse_score"]
                agent_payload["delta_detection"] = {
                    "prior_score": prior["adverse_score"],
                    "prior_date": prior["timestamp"],
                    "score_delta": round(score_delta, 2),
                }
            store.close()
        except Exception:
            pass  # Skip persistence for incomplete/error payloads — never mask a successful score

        return json.dumps(agent_payload)

    except Exception as e:
        log_event(logger, "tool_error", error=str(e))
        error_payload = {
            'metadata': {
                'tool_name': 'AdverseScore',
                'status': 'System Error',
                'clinical_disclaimer': 'This tool is for informational purposes only and does not constitute medical advice.'
            },
            'agent_directives': {
                'diagnosis_lock': True,
                'requires_human_review': False,
                'route_to_specialist': False,
                'system_directive': 'Inform the user that the AdverseScore tool encountered a system error and could not complete the analysis. Please try again or contact support.'
            }
        }
        return json.dumps(error_payload)
    
