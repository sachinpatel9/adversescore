import sys
import os
import re
from pathlib import Path

#pointing python to 'srs' directory
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
import streamlit.components.v1 as components
from adverse_score.orchestrator import agent_executor
from langchain_core.messages import HumanMessage, AIMessage

NARRATIVE_KEYWORDS = {"write", "narrative", "document", "report", "summarise", "summarize", "memo", "draft"}

def message_requests_narrative(text: str) -> bool:
    """Check if a user message contains documentation-intent keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in NARRATIVE_KEYWORDS)

#UI Configuration
st.set_page_config(page_title="AdverseScore Clinical AI", page_icon="⚕️", layout='centered')

#Styling — Design System
st.markdown("""
    <style>
    /* ── Design System Variables ── */
    :root {
        --color-primary: #1a73e8;
        --color-bg: #ffffff;
        --color-text: #1f2937;
        --color-text-muted: #6b7280;
        --color-border: #e5e7eb;
        --color-surface: #f8f9fa;
        --font-family: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        --radius: 8px;
        --shadow-card: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    }

    /* ── Strip ALL Streamlit branding ── */
    #MainMenu, footer, header,
    .stDeployButton,
    [data-testid="stDecoration"],
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* ── Global Typography ── */
    html, body, [class*="css"] {
        font-family: var(--font-family) !important;
    }

    /* ── Page Layout ── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 780px;
    }

    /* ── Branded Header Bar ── */
    .header-bar {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--color-border);
        margin-bottom: 1.5rem;
    }
    .header-bar .logo {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--color-text);
        letter-spacing: -0.02em;
    }
    .header-bar .logo-accent {
        color: var(--color-primary);
    }
    .header-bar .tagline {
        font-size: 0.78rem;
        color: var(--color-text-muted);
        margin-left: auto;
    }

    /* ── Chat Styling ── */
    [data-testid="stChatMessage"] {
        border-radius: var(--radius);
    }
    </style>
""", unsafe_allow_html=True)

# Branded header bar
st.markdown("""
<div class="header-bar">
    <div class="logo"><span class="logo-accent">Adverse</span>Score</div>
    <div class="tagline">Clinical Decision Support &middot; v1.0</div>
</div>
""", unsafe_allow_html=True)

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
                # Build history from session state. The current prompt was already
                # appended to st.session_state.messages above (line 103), so the
                # loop below includes it — no need to append it again.
                history = []
                for msg in st.session_state.messages:
                    if msg['role'] == 'user':
                        history.append(HumanMessage(content=msg['content']))
                    elif msg['role'] == 'assistant':
                        history.append(AIMessage(content=msg['content']))
                response = agent_executor.invoke(
                    {'messages': history},
                    config={'recursion_limit': 10}
                )
                output = response['messages'][-1].content
                # Extract Signal Evaluation Narrative if present
                narrative_match = re.search(
                    r'<!-- SIGNAL_NARRATIVE_START -->\s*(.*?)\s*<!-- SIGNAL_NARRATIVE_END -->',
                    output, re.DOTALL
                )
                narrative_text = None
                if narrative_match and message_requests_narrative(prompt):
                    narrative_text = narrative_match.group(1).strip()
                    output = output[:narrative_match.start()].rstrip() + output[narrative_match.end():].lstrip('\n')
                # Extract Score Rationale section for expander rendering
                rationale_match = re.search(
                    r'(?:#{1,3}\s*\**Score Rationale\**|(?:\d+\.\s*)?\**Score Rationale\**)'
                    r'[:\s—-]*\n(.*?)(?=\n#{1,3}\s|\n\d+\.\s\**[A-Z]|\Z)',
                    output, re.DOTALL | re.IGNORECASE
                )
                if rationale_match:
                    rationale_text = rationale_match.group(1).strip()
                    # Render main output without the rationale section
                    main_output = output[:rationale_match.start()].rstrip()
                    remainder = output[rationale_match.end():].lstrip('\n')
                    st.markdown(main_output)
                    with st.expander("Score Rationale", expanded=True):
                        st.markdown(rationale_text)
                    if remainder:
                        st.markdown(remainder)
                else:
                    st.markdown(output)
                # Render label status badge based on LLM response content
                output_upper = output.upper()
                if "UNLABELED" in output_upper:
                    st.warning("UNLABELED — This adverse event is not in the official drug label")
                elif "LABEL_STATUS_UNKNOWN" in output_upper:
                    st.info("Label status could not be determined from FDA data")

                # Render Signal Evaluation Narrative in expander with download/copy
                if narrative_text:
                    with st.expander("Signal Evaluation Narrative — Draft", expanded=False):
                        st.markdown(narrative_text)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "Download as .txt",
                                data=narrative_text,
                                file_name="signal_narrative.txt",
                                mime="text/plain"
                            )
                        with col2:
                            escaped = narrative_text.replace('`', '\\`').replace('\\', '\\\\')
                            components.html(
                                f"""<button onclick="navigator.clipboard.writeText(`{escaped}`).then(()=>this.textContent='Copied!')"
                                style="padding:0.4em 1em;border:1px solid #ccc;border-radius:4px;cursor:pointer;background:#fff;font-family:sans-serif;font-size:14px;">
                                Copy to clipboard</button>""",
                                height=45
                            )

                st.session_state.messages.append({'role': 'assistant', 'content': output})
            except Exception as e:
                st.error(f"System Error: {str(e)}")