import sys
from pathlib import Path

#pointing python to 'srs' directory
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
from adverse_score.orchestrator import agent_executor

# ── UI Configuration ──────────────────────────────────────────────────────────

st.set_page_config(page_title="AdverseScore Clinical AI", page_icon="⚕️", layout='wide')

# ── Design System ─────────────────────────────────────────────────────────────

_dark = st.session_state.get("dark_mode", False)

st.markdown("""
<style>
/* ── Design System Variables ── */
:root {
    --color-primary: #1a73e8;
    --color-primary-light: #e8f0fe;
    --color-success: #059669;
    --color-success-light: #d1fae5;
    --color-warning: #d97706;
    --color-warning-light: #fef3c7;
    --color-danger: #dc2626;
    --color-danger-light: #fee2e2;
    --color-neutral: #6b7280;
    --color-neutral-light: #f3f4f6;
    --color-bg-page: #f9fafb;
    --color-surface: #ffffff;
    --color-text: #1f2937;
    --color-text-muted: #6b7280;
    --color-text-secondary: #9ca3af;
    --color-border: #e5e7eb;
    --color-border-light: #f3f4f6;
    --font-family: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --text-xs: 0.7rem;
    --text-sm: 0.8rem;
    --text-base: 0.9rem;
    --text-lg: 1.05rem;
    --text-xl: 1.25rem;
    --text-2xl: 1.6rem;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --radius-full: 100px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-lg: 0 4px 16px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
    --space-xs: 0.25rem;
    --space-sm: 0.5rem;
    --space-md: 0.75rem;
    --space-lg: 1rem;
    --space-xl: 1.5rem;
    --space-2xl: 2rem;
    --space-3xl: 3rem;
}

/* ── Strip Streamlit Branding ── */
#MainMenu, footer, header,
.stDeployButton,
[data-testid="stDecoration"],
[data-testid="stHeader"] {
    display: none !important;
}

/* ── Global Typography ── */
html, body, [class*="css"] {
    font-family: var(--font-family) !important;
    color: var(--color-text);
}

/* ── Page Layout ── */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
    margin: 0 auto;
}

/* ── Card System ── */
.card {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: var(--space-lg);
    box-shadow: var(--shadow-sm);
    margin-bottom: var(--space-lg);
}
.card-elevated {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: var(--space-xl);
    box-shadow: var(--shadow-md);
    margin-bottom: var(--space-lg);
}
.card-accent-danger { border-left: 4px solid var(--color-danger); }
.card-accent-warning { border-left: 4px solid var(--color-warning); }
.card-accent-success { border-left: 4px solid var(--color-success); }
.card-accent-primary { border-left: 4px solid var(--color-primary); }

/* ── Badge System ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    padding: 0.2em 0.65em;
    border-radius: var(--radius-full);
    font-size: var(--text-xs);
    font-weight: 600;
    letter-spacing: 0.02em;
    line-height: 1.4;
    white-space: nowrap;
}
.badge-danger { background: var(--color-danger-light); color: var(--color-danger); }
.badge-warning { background: var(--color-warning-light); color: var(--color-warning); }
.badge-success { background: var(--color-success-light); color: var(--color-success); }
.badge-neutral { background: var(--color-neutral-light); color: var(--color-neutral); }
.badge-primary { background: var(--color-primary-light); color: var(--color-primary); }

/* ── Header Bar ── */
.header-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 0 0.75rem 0;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, var(--color-primary) 0%, transparent 60%) 1;
    margin-bottom: var(--space-xl);
}
.header-bar .logo {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--color-text);
    letter-spacing: -0.03em;
}
.header-bar .logo-accent { color: var(--color-primary); }
.header-bar .tagline {
    font-size: var(--text-sm);
    color: var(--color-text-muted);
    margin-left: auto;
}

/* ── Chat Message Overrides ── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-md);
    padding: var(--space-md) var(--space-lg);
    margin-bottom: var(--space-sm);
    max-width: 900px;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .block-container { max-width: 100%; }
}
</style>
""", unsafe_allow_html=True)

# ── Dark Mode CSS Overrides ───────────────────────────────────────────────────
if _dark:
    st.markdown("""
    <style>
    :root {
        --color-bg-page: #0f1117;
        --color-surface: #1e1e2e;
        --color-text: #e5e7eb;
        --color-text-muted: #9ca3af;
        --color-text-secondary: #6b7280;
        --color-border: #374151;
        --color-border-light: #1f2937;
        --color-neutral-light: #1f2937;
        --color-primary-light: #1e3a5f;
        --color-success-light: #064e3b;
        --color-warning-light: #78350f;
        --color-danger-light: #7f1d1d;
    }
    [data-testid="stSidebar"] { background: #1a1a2e; }
    </style>
    """, unsafe_allow_html=True)

# ── Branded Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div class="logo"><span class="logo-accent">Adverse</span>Score</div>
    <span class="badge badge-neutral">v1.0</span>
    <div class="tagline">PSUR Consolidation — Clinical Decision Support</div>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ── Chat History Display ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# ── Execution Loop (placeholder pending Phase 8 agent rebuild) ────────────────
if prompt := st.chat_input('Analyze a drug safety profile....'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        if agent_executor is None:
            placeholder_response = (
                "The PSUR consolidation agent is being rebuilt "
                "(see docs/PSUR_CONSOLIDATION_SCOPE.md, Phase 8). "
                "This chat is a placeholder until the new agent orchestration lands."
            )
            st.info(placeholder_response)
            st.session_state.messages.append({'role': 'assistant', 'content': placeholder_response})
