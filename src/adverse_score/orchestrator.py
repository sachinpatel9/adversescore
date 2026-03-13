import json 
from langchain_core.tools import tool
from .client import AdverseScoreClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from .config import initialize_config

#instantiate the client globally so the HTTP session persists across multiple agent calls
_global_client = AdverseScoreClient()

#LangChain Tool Wrapper
@tool
def get_adverse_score(drug_name: str) -> str:
    """
    calculates a clinical safety risk score (0-100) and extracts relative reporting rates
    for a specified pharmaceutical drug based on the FDA adverse event reports
    Call this tool when a user asks about the safety, side effects, toxicity, or risk profile of a specific medication.
    """
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

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.0,
)

#define the prompt 
system_instructions = """
You are a Clinical Decision Support AI. Your primary function is to assist researchers and clinicians by analyzing pharmaceutical safety signals using the `get_adverse_score` tool. You act as an analytical interface, not a physician.

CRITICAL SAFETY PROTOCOLS:
When you receive a JSON payload from the `get_adverse_score` tool, you must parse the `agent_directives` object and strictly obey the following rules:

1. THE DIAGNOSIS LOCK: If `diagnosis_lock` is true, you MUST NOT formulate a diagnosis, recommend changes to a patient's medication, or offer autonomous medical advice. You are restricted to summarizing the statistical risk data, confidence intervals, and peer benchmarks provided in the payload.
2. HUMAN-IN-THE-LOOP: If `requires_human_review` is true, you MUST explicitly state that the adverse signal is dangerously high (or that the data quality is too low to be trusted). You must immediately halt further autonomous analysis and instruct the user to escalate the findings to a human clinical safety officer.
3. SPECIALIST ROUTING: If `route_to_specialist` is true, explicitly flag the output for specialized review (e.g., an Oncologist for high-toxicity therapies).
4. SYSTEM DIRECTIVES: If the payload contains a specific `system_directive` string, you must execute that instruction as your highest priority.
5. THE DISCLAIMER: You must invariably append the `clinical_disclaimer` from the payload metadata to your final response to the user.

TONE & STYLE:
Be objective, strictly factual, and concise. Do not use alarming or sensational language. Always present the `relative_risk` and `class_benchmark_avg` to ensure the user has the proper context for the target drug's performance.

"""

#group the tools
tools = [get_adverse_score]

#create the executor
agent_executor = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt=system_instructions
)

#Interactive Chat Loop
if __name__ == "__main__":
    print("\n" + "="*50)
    print("ADVERSESCORE CLINICAL AGENT ONLINE")
    print("Type 'exit' or 'quit' to terminate the session.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ['exit', 'quit']:
                print('\nTerminating session. goodbye!')
                break
            if not user_input.strip():
                continue

            #execute the agent pipeline
            response = agent_executor.invoke({
                'messages': [HumanMessage(content=user_input)]
            })

            print("\n" + "-"*50)
            print(f"Agent:\n{response['messages'][-1].content}")
            print("-"*50)

        except KeyboardInterrupt:
            print("\nTerminating session. Goodbye.")
            break
        except Exception as e:
            print(f"\n[System Error]: {e}")
