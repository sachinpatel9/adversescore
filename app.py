import sys
import os
from pathlib import Path

#pointing python to 'srs' directory
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
from adverse_score.orchestrator import agent_executor
from langchain_core.messages import HumanMessage

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
                #triggering the LangGraph agent
                response = agent_executor.invoke({'messages': [HumanMessage(content=prompt)]})
                output = response['messages'][-1].content
                st.markdown(output)
                st.session_state.messages.append({'role': 'assistant', 'content': output})
            except Exception as e:
                st.error(f"System Error: {str(e)}")