import sys
import os
from pathlib import Path

#pointing python to 'srs' directory
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
from adverse_score.orchestrator import agent_executor
from langchain_core.messages import HumanMessage, AIMessage

#UI Configuration
st.set_page_config(page_title="AdverseScore Clinical AI", page_icon="⚕️", layout='centered')

#Styling
st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Vertically center the main content and constrain width */
    .block-container {
        padding-top: 10vh;
        padding-bottom: 5vh;
        max-width: 750px;
    }
    </style>
""", unsafe_allow_html=True)

# typographic centering 
# Using HTML instead of st.title to force center alignment
st.markdown("<h1 style='text-align: center;'>⚕️ AdverseScore</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6c757d; margin-top: -10px;'>Clinical Decision Support Agent</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 0.85em; color: gray;'>v1.0-MVP | Powered by openFDA & LangGraph</p>", unsafe_allow_html=True)
st.markdown("---")

#session state for persistent conversation
if 'messages' not in st.session_state:
    st.session_state.messages = []

#Display history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

#Execution Loop 
if prompt := st.chat_input('Analyze a drug safety profile....'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    
    with st.chat_message('assistant'):
        with st.spinner('Executing clinical scoring and peer benchmarking...'):
            try:
                # FIX: The default recursion_limit for create_agent is 10,000 — effectively
                # unlimited. Each LLM→tool→LLM cycle uses ~3 graph steps, so 10,000 steps
                # allows ~3,333 tool calls. For this agent (single tool, one call per request),
                # recursion_limit=10 allows up to ~3 tool calls which covers the normal flow
                # (1 call) plus buffer for edge cases, while preventing runaway loops that
                # would burn API credits and hang the UI indefinitely.

                # FIX: The old code sent only the current message, making the agent
                # stateless — it couldn't handle follow-ups like "now compare to
                # ibuprofen" because it had no memory of the previous drug analyzed.
                # Now we pass the full conversation history so the agent can reference
                # prior turns. The system prompt's SCOPE ENFORCEMENT and TOOL PROTOCOL
                # rules still apply per-turn.
                history = []
                for msg in st.session_state.messages:
                    if msg['role'] == 'user':
                        history.append(HumanMessage(content=msg['content']))
                    elif msg['role'] == 'assistant':
                        history.append(AIMessage(content=msg['content']))
                history.append(HumanMessage(content=prompt))
                response = agent_executor.invoke(
                    {'messages': history},
                    config={'recursion_limit': 10}
                )
                output = response['messages'][-1].content
                st.markdown(output)
                st.session_state.messages.append({'role': 'assistant', 'content': output})
            except Exception as e:
                st.error(f"System Error: {str(e)}")