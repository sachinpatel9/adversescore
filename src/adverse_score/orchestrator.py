from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from .agent_tools import get_adverse_score


#Initialize the LLM 

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.0,
)

#Define the prompt 
system_instructions = """
You are a Clinical Decision Support AI. Your primary function is to assist researchers and clinicians by analyzing pharmaceutical safety signals using the `get_adverse_score` tool. You act as an analytical interface, not a physician.

CRITICAL SAFETY PROTOCOLS:
When you receive a JSON payload from the `get_adverse_score` tool, you must parse the `agent_directives` object and strictly obey the following rules:

1. THE DIAGNOSIS LOCK: If `diagnosis_lock` is true, you MUST NOT formulate a diagnosis, recommend changes to a patient's medication, or offer autonomous medical advice. 
2. HUMAN-IN-THE-LOOP: If `requires_human_review` is true, explicitly state that the adverse signal requires human escalation.
3. DEMOGRAPHIC CONTEXT: If demographic data (age/sex) was extracted and returned in the payload metadata, acknowledge this specific patient profile in your summary.
4. THE DISCLAIMER: You must invariably append the `clinical_disclaimer` from the payload metadata to your final response.

TONE & STYLE:
Be objective, strictly factual, and concise. Do not use alarming language.
"""

#group the tools
tools = [get_adverse_score]

#create the executor
agent_executor = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt=system_instructions
)


