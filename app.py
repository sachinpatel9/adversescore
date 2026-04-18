import sys
import os
import re
import json
import html as html_mod
from pathlib import Path

#pointing python to 'srs' directory
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from adverse_score.orchestrator import agent_executor
from adverse_score.persistence import AnalysisStore
from langchain_core.messages import HumanMessage, AIMessage

# ── Keyword Sets ──────────────────────────────────────────────────────────────

NARRATIVE_KEYWORDS = {"write", "narrative", "document", "report", "summarise", "summarize", "memo", "draft"}

def message_requests_narrative(text: str) -> bool:
    """Check if a user message contains documentation-intent keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in NARRATIVE_KEYWORDS)

TEMPORAL_KEYWORDS = {"trend", "over time", "quarterly", "changing", "getting worse", "getting better", "historical", "last quarter", "recent quarters"}

def message_requests_temporal(text: str) -> bool:
    """Check if a user message contains temporal analysis keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in TEMPORAL_KEYWORDS)


# ── Helper: Score Color ───────────────────────────────────────────────────────

def _score_color(score):
    """Return CSS color class suffix based on score threshold."""
    if score <= 30:
        return "success"
    elif score <= 70:
        return "warning"
    return "danger"

def _score_hex(score):
    """Return hex color for score value."""
    if score <= 30:
        return "#059669"
    elif score <= 70:
        return "#d97706"
    return "#dc2626"

def _trend_badge_html(classification):
    """Return badge HTML for a trend classification."""
    mapping = {
        "RISING": ("danger", "Rising"),
        "STABLE": ("neutral", "Stable"),
        "DECLINING": ("success", "Declining"),
        "INSUFFICIENT_DATA": ("neutral", "Insufficient Data"),
    }
    variant, label = mapping.get(classification, ("neutral", classification or "—"))
    return f'<span class="badge badge-{variant}">{label}</span>'


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

/* ── Score Gauge ── */
.score-gauge {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    margin-top: var(--space-xs);
}
.score-bar-bg {
    flex: 1;
    height: 8px;
    background: var(--color-neutral-light);
    border-radius: 4px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}
.score-bar-fill.score-low { background: var(--color-success); }
.score-bar-fill.score-mid { background: var(--color-warning); }
.score-bar-fill.score-high { background: var(--color-danger); }

/* ── Metric Display ── */
.metric-value {
    font-size: var(--text-2xl);
    font-weight: 700;
    line-height: 1.1;
    margin: 0;
}
.metric-label {
    font-size: var(--text-xs);
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

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

/* ── Hero Card ── */
.hero-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: var(--space-lg);
    align-items: start;
}
.hero-drug-name {
    font-size: var(--text-xl);
    font-weight: 700;
    margin: 0 0 var(--space-xs) 0;
    color: var(--color-text);
}
.hero-status {
    font-size: var(--text-sm);
    color: var(--color-text-muted);
    margin-top: var(--space-xs);
}
.hero-metric-block { text-align: center; }

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: var(--color-surface);
    border-right: 1px solid var(--color-border-light);
}
[data-testid="stSidebar"] .stButton > button {
    text-align: left;
    background: var(--color-bg-page);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.75rem;
    font-size: var(--text-sm);
    color: var(--color-text);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    margin-bottom: 0.25rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--color-primary);
    box-shadow: var(--shadow-sm);
}
.sidebar-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: var(--space-md);
}
.sidebar-header h3 {
    margin: 0;
    font-size: var(--text-lg);
    font-weight: 700;
}
.sidebar-count {
    background: var(--color-primary-light);
    color: var(--color-primary);
    font-size: var(--text-xs);
    font-weight: 600;
    padding: 0.15em 0.5em;
    border-radius: var(--radius-full);
}
.empty-state {
    text-align: center;
    padding: var(--space-2xl) var(--space-lg);
    color: var(--color-text-muted);
}
.empty-state-icon {
    font-size: 2rem;
    margin-bottom: var(--space-sm);
    opacity: 0.4;
}
.empty-state-text {
    font-size: var(--text-sm);
}

/* ── Chat Message Overrides ── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-md);
    padding: var(--space-md) var(--space-lg);
    margin-bottom: var(--space-sm);
    max-width: 900px;
}

/* ── Expander Overrides ── */
[data-testid="stExpander"] {
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
    margin: var(--space-md) 0;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    font-size: var(--text-base);
}

/* ── Scorecard Styling ── */
.scorecard-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--space-lg);
}
.scorecard-header h3 { margin: 0; font-weight: 700; }
.scorecard-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: var(--space-md);
    margin-bottom: var(--space-xl);
}
.scorecard-metric-card {
    background: var(--color-bg-page);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-sm);
    padding: var(--space-md) var(--space-lg);
    text-align: center;
}

/* ── Trend Badge ── */
.trend-badge-row {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    margin-top: var(--space-sm);
    margin-bottom: var(--space-md);
}

/* ── Error Card ── */
.error-card {
    background: var(--color-surface);
    border: 1px solid var(--color-danger-light);
    border-left: 4px solid var(--color-danger);
    border-radius: var(--radius-md);
    padding: var(--space-lg);
}
.error-card-header {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    margin-bottom: var(--space-sm);
}
.error-card-header span { font-weight: 600; }
.error-card p { color: var(--color-text-muted); margin: 0; font-size: var(--text-sm); }

/* ── Copy Button Styling ── */
.copy-btn {
    padding: 0.4em 1em;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    background: var(--color-surface);
    font-family: var(--font-family);
    font-size: var(--text-sm);
    color: var(--color-text);
    transition: border-color 0.15s, background 0.15s;
}
.copy-btn:hover { border-color: var(--color-primary); background: var(--color-primary-light); }

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero-grid { grid-template-columns: 1fr 1fr; }
    .scorecard-metrics { grid-template-columns: 1fr 1fr; }
    .block-container { max-width: 100%; }
}
@media (max-width: 480px) {
    .hero-grid { grid-template-columns: 1fr; }
    .scorecard-metrics { grid-template-columns: 1fr; }
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
    <div class="tagline">Clinical Decision Support</div>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ── Sidebar: Analysis History ─────────────────────────────────────────────────
with st.sidebar:
    # Dark mode toggle at top
    st.toggle("Dark mode", value=False, key="dark_mode")
    st.markdown("---")

    _store = AnalysisStore()
    _history = _store.get_history(limit=20)
    _store.close()

    _count = len(_history)
    st.markdown(f"""
    <div class="sidebar-header">
        <h3>Analysis History</h3>
        <span class="sidebar-count">{_count}</span>
    </div>
    """, unsafe_allow_html=True)

    # Search filter
    _search = st.text_input("", placeholder="Filter drugs...", label_visibility="collapsed")
    if _search:
        _history = [r for r in _history if _search.lower() in r["drug_name"].lower()]

    if not _history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">&#128202;</div>
            <div class="empty-state-text">No analyses yet.<br>Run a drug query to begin.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sort_by = st.radio("Sort by", ["Recent", "Drug Name", "Score"],
                           horizontal=True, label_visibility="collapsed")
        if sort_by == "Drug Name":
            _history.sort(key=lambda r: r["drug_name"])
        elif sort_by == "Score":
            _history.sort(key=lambda r: r["adverse_score"], reverse=True)

        for _row in _history:
            _date_short = _row["timestamp"][:10]
            _s = _row["adverse_score"]
            _color_cls = _score_color(_s)
            if st.button(
                f"{_row['drug_name']}  —  {_s:.0f}  —  {_date_short}",
                key=f"hist_{_row['id']}",
                use_container_width=True
            ):
                st.session_state["prefill_query"] = f"Analyze {_row['drug_name']}"
                st.rerun()

    st.markdown("---")
    if st.button("Compare Portfolio", type="primary", use_container_width=True):
        st.session_state["show_scorecard"] = True
        st.rerun()

# ── Comparative Safety Scorecard ──────────────────────────────────────────────
if st.session_state.get("show_scorecard"):
    _store = AnalysisStore()
    _portfolio = _store.get_portfolio()
    _store.close()

    sc_col1, sc_col2 = st.columns([10, 1])
    with sc_col1:
        st.markdown('<div class="scorecard-header"><h3>Comparative Safety Scorecard</h3></div>', unsafe_allow_html=True)
    with sc_col2:
        if st.button("✕", key="close_scorecard"):
            st.session_state["show_scorecard"] = False
            st.rerun()

    if not _portfolio:
        st.info("No analyses available for comparison.")
    else:
        # Summary metrics row
        _total = len(_portfolio)
        _scores = [p["adverse_score"] for p in _portfolio]
        _avg = sum(_scores) / _total if _total else 0
        _highest = max(_portfolio, key=lambda p: p["adverse_score"])
        _rising = sum(1 for p in _portfolio if p.get("trend_classification") == "RISING")

        st.markdown(f"""
        <div class="scorecard-metrics">
            <div class="scorecard-metric-card">
                <div class="metric-value">{_total}</div>
                <div class="metric-label">Drugs Analyzed</div>
            </div>
            <div class="scorecard-metric-card">
                <div class="metric-value" style="color: {_score_hex(_highest['adverse_score'])};">{_highest['drug_name']}</div>
                <div class="metric-label">Highest Risk ({_highest['adverse_score']:.0f})</div>
            </div>
            <div class="scorecard-metric-card">
                <div class="metric-value">{_avg:.0f}</div>
                <div class="metric-label">Avg Score</div>
            </div>
            <div class="scorecard-metric-card">
                <div class="metric-value" style="color: var(--color-danger);">{_rising}</div>
                <div class="metric-label">Rising Trends</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced table
        _df = pd.DataFrame(_portfolio)
        _df = _df[["drug_name", "adverse_score", "prr_value", "confidence_level",
                    "trend_classification", "label_status"]]
        _df.columns = ["Drug", "AdverseScore", "PRR", "Confidence", "Trend", "Label Status"]
        _df["PRR"] = _df["PRR"].apply(lambda v: f"{v:.2f}" if v is not None else "N/A")
        _df["Trend"] = _df["Trend"].fillna("—")
        _df["Label Status"] = _df["Label Status"].fillna("—")

        def _style_scorecard(styler):
            styler.background_gradient(
                subset=["AdverseScore"], cmap="RdYlGn_r", vmin=0, vmax=100
            )
            return styler

        styled = _df.style.pipe(_style_scorecard).format({"AdverseScore": "{:.0f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

# ── Chat History Display ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# ── Execution Loop ────────────────────────────────────────────────────────────
_prefill = st.session_state.pop("prefill_query", None)
if _prefill:
    prompt = _prefill
elif prompt := st.chat_input('Analyze a drug safety profile....'):
    pass
else:
    prompt = None

if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.status("Analyzing safety profile...", expanded=True) as status:
            try:
                status.write("Querying FDA adverse event database...")

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

                # Extract the analyzed drug name from the tool payload so the narrative
                # save block has a reliable source that is immune to DB race conditions.
                st.session_state["last_analyzed_drug"] = None
                for _msg in response['messages']:
                    _content = getattr(_msg, 'content', '')
                    if isinstance(_content, str) and '"drug_target"' in _content:
                        try:
                            _payload = json.loads(_content)
                            _drug = _payload.get('clinical_signal', {}).get('drug_target')
                            if _drug:
                                st.session_state["last_analyzed_drug"] = _drug
                                break
                        except (json.JSONDecodeError, AttributeError):
                            pass

                status.update(label="Analysis complete", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Analysis failed", state="error", expanded=False)
                _safe_err = html_mod.escape(str(e))
                st.markdown(f"""
                <div class="error-card">
                    <div class="error-card-header">
                        <span>Analysis Error</span>
                        <span class="badge badge-danger">System Error</span>
                    </div>
                    <p>{_safe_err}</p>
                </div>
                """, unsafe_allow_html=True)
                output = None

        if output:
            # ── Extract structured data from LLM response ─────────────────

            # 1. Extract Signal Evaluation Narrative
            narrative_match = re.search(
                r'<!-- SIGNAL_NARRATIVE_START -->\s*(.*?)\s*<!-- SIGNAL_NARRATIVE_END -->',
                output, re.DOTALL
            )
            narrative_text = None
            if narrative_match and message_requests_narrative(prompt):
                narrative_text = narrative_match.group(1).strip()
                output = output[:narrative_match.start()].rstrip() + output[narrative_match.end():].lstrip('\n')
                _analyzed_drug = st.session_state.get("last_analyzed_drug")
                if _analyzed_drug:
                    with AnalysisStore() as _ns:
                        _ns.update_narrative(_analyzed_drug, narrative_text)

            # 2. Extract time_series JSON
            time_series_match = re.search(
                r'<!-- TIME_SERIES_DATA_START -->\s*(.*?)\s*<!-- TIME_SERIES_DATA_END -->',
                output, re.DOTALL
            )
            time_series_data = None
            if time_series_match and message_requests_temporal(prompt):
                try:
                    time_series_data = json.loads(time_series_match.group(1).strip())
                except json.JSONDecodeError:
                    time_series_data = None
                output = output[:time_series_match.start()].rstrip() + output[time_series_match.end():].lstrip('\n')

            # 3. Extract Score Rationale
            rationale_match = re.search(
                r'(?:#{1,3}\s*\**Score Rationale\**|(?:\d+\.\s*)?\**Score Rationale\**)'
                r'[:\s—-]*\n(.*?)(?=\n#{1,3}\s|\n\d+\.\s\**[A-Z]|\Z)',
                output, re.DOTALL | re.IGNORECASE
            )

            # 4. Extract score value for hero card
            score_match = re.search(r'AdverseScore[:\s]*(\d+(?:\.\d+)?)\s*/\s*100', output, re.IGNORECASE)
            drug_match = re.search(r'\*\*([A-Z][A-Z0-9\s\-/]+?)\*\*', output)
            prr_match = re.search(r'PRR[:\s]*(\d+(?:\.\d+)?)', output, re.IGNORECASE)

            # ── Render Score Hero Card ────────────────────────────────────
            if score_match:
                _score_val = float(score_match.group(1))
                _drug_name = html_mod.escape(drug_match.group(1).strip()) if drug_match else "Drug"
                _prr_val = float(prr_match.group(1)) if prr_match else None
                _gauge_cls = "score-low" if _score_val <= 30 else ("score-mid" if _score_val <= 70 else "score-high")
                _score_color_hex = _score_hex(_score_val)

                _prr_html = ""
                if _prr_val is not None:
                    _prr_html = f"""
                    <div class="hero-metric-block">
                        <div class="metric-value">{_prr_val:.2f}</div>
                        <div class="metric-label">PRR</div>
                    </div>"""
                else:
                    _prr_html = """
                    <div class="hero-metric-block">
                        <div class="metric-value" style="color: var(--color-text-secondary);">—</div>
                        <div class="metric-label">PRR</div>
                    </div>"""

                st.markdown(f"""
                <div class="card-elevated">
                    <div class="hero-grid">
                        <div>
                            <div class="hero-drug-name">{_drug_name}</div>
                            <div class="score-gauge">
                                <div class="score-bar-bg">
                                    <div class="score-bar-fill {_gauge_cls}" style="width: {min(_score_val, 100):.0f}%;"></div>
                                </div>
                            </div>
                        </div>
                        <div class="hero-metric-block">
                            <div class="metric-value" style="color: {_score_color_hex};">{_score_val:.0f}</div>
                            <div class="metric-label">AdverseScore</div>
                        </div>
                        {_prr_html}
                        <div class="hero-metric-block">
                            <span class="badge badge-{_score_color(_score_val)}">{
                                'Low Risk' if _score_val <= 30 else ('Moderate' if _score_val <= 70 else 'High Risk')
                            }</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Render main response ──────────────────────────────────────
            if rationale_match:
                rationale_text = rationale_match.group(1).strip()
                main_output = output[:rationale_match.start()].rstrip()
                remainder = output[rationale_match.end():].lstrip('\n')
                st.markdown(main_output)
                with st.expander("Score Rationale", expanded=True):
                    st.markdown(rationale_text)
                if remainder:
                    st.markdown(remainder)
            else:
                st.markdown(output)

            # ── Label status badges (styled) ──────────────────────────────
            output_upper = output.upper()
            if "UNLABELED" in output_upper:
                st.markdown("""
                <div class="trend-badge-row">
                    <span class="badge badge-danger">&#9888; UNLABELED</span>
                    <span style="font-size: var(--text-sm); color: var(--color-text-muted);">This adverse event is not in the official drug label</span>
                </div>
                """, unsafe_allow_html=True)
            elif "LABEL_STATUS_UNKNOWN" in output_upper:
                st.markdown("""
                <div class="trend-badge-row">
                    <span class="badge badge-neutral">? Unknown</span>
                    <span style="font-size: var(--text-sm); color: var(--color-text-muted);">Label status could not be determined from FDA data</span>
                </div>
                """, unsafe_allow_html=True)

            # ── Trend chart (enhanced Plotly) ─────────────────────────────
            if time_series_data and len(time_series_data) >= 2:
                quarters = [d["quarter"] for d in time_series_data]
                scores = [d["adverse_score"] for d in time_series_data]
                prr_values = [d.get("prr") for d in time_series_data]

                _chart_font = "'Inter', 'DM Sans', -apple-system, system-ui, sans-serif"
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=quarters, y=scores, mode='lines+markers',
                    name='AdverseScore',
                    line=dict(color='#1a73e8', width=2.5),
                    marker=dict(size=8, line=dict(width=2, color='white'))
                ))
                if any(v is not None for v in prr_values):
                    fig.add_trace(go.Scatter(
                        x=quarters, y=[v if v is not None else 0 for v in prr_values],
                        mode='lines+markers', name='PRR', yaxis='y2',
                        line=dict(color='#e8711a', width=2, dash='dot'),
                        marker=dict(size=6, symbol='diamond', line=dict(width=1, color='white'))
                    ))
                    fig.update_layout(yaxis2=dict(
                        title=dict(text='PRR', font=dict(family=_chart_font, size=12, color='#6b7280')),
                        overlaying='y', side='right', showgrid=False,
                        tickfont=dict(family=_chart_font, size=11, color='#9ca3af')
                    ))
                fig.update_layout(
                    title=dict(text='Quarterly Trend Analysis', font=dict(family=_chart_font, size=15, color='#1f2937')),
                    xaxis=dict(
                        title=dict(text='Quarter', font=dict(family=_chart_font, size=12, color='#6b7280')),
                        tickfont=dict(family=_chart_font, size=11, color='#9ca3af'),
                        gridcolor='#f3f4f6', gridwidth=1
                    ),
                    yaxis=dict(
                        title=dict(text='AdverseScore (0-100)', font=dict(family=_chart_font, size=12, color='#6b7280')),
                        tickfont=dict(family=_chart_font, size=11, color='#9ca3af'),
                        gridcolor='#f3f4f6', gridwidth=1, griddash='dot'
                    ),
                    template='plotly_white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=380, margin=dict(t=50, b=40, l=50, r=50),
                    legend=dict(
                        orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1,
                        font=dict(family=_chart_font, size=11, color='#6b7280'),
                        bgcolor='rgba(255,255,255,0.8)', bordercolor='#e5e7eb', borderwidth=1
                    ),
                    hoverlabel=dict(font=dict(family=_chart_font))
                )
                st.plotly_chart(fig, use_container_width=True)

                # Trend badge
                delta = scores[-1] - (scores[-3] if len(scores) >= 3 else scores[0])
                if delta >= 10:
                    _trend_cls, _trend_label = "danger", "RISING — AdverseScore increased by 10+ points over recent quarters"
                elif delta <= -10:
                    _trend_cls, _trend_label = "success", "DECLINING — AdverseScore decreased by 10+ points over recent quarters"
                else:
                    _trend_cls, _trend_label = "neutral", "STABLE — AdverseScore remained within 10 points over recent quarters"
                st.markdown(f"""
                <div class="trend-badge-row">
                    <span class="badge badge-{_trend_cls}">{_trend_label.split(' — ')[0]}</span>
                    <span style="font-size: var(--text-sm); color: var(--color-text-muted);">{_trend_label.split(' — ')[1]}</span>
                </div>
                """, unsafe_allow_html=True)

            # ── Signal Narrative expander ──────────────────────────────────
            if narrative_text:
                with st.expander("Signal Evaluation Narrative — Draft", expanded=False):
                    st.markdown('<span class="badge badge-primary">DRAFT — For Human Review Only</span>', unsafe_allow_html=True)
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
                        escaped = narrative_text.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
                        components.html(
                            f"""<button class="copy-btn" onclick="navigator.clipboard.writeText(`{escaped}`).then(()=>{{this.textContent='Copied!';this.style.borderColor='#059669';this.style.color='#059669';}})">
                            Copy to clipboard</button>""",
                            height=45
                        )

            st.session_state.messages.append({'role': 'assistant', 'content': output})
