import html
import io
import sys
import uuid
from pathlib import Path

#pointing python to 'srs' directory
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st

from adverse_score.config import TOP_N_NARRATED_SIGNALS
from adverse_score.consolidation import ConsolidationError, ConsolidationResult, consolidate_psur
from adverse_score.document_generator import generate_psur_document
from adverse_score.orchestrator import run_agent_turn
from adverse_score.persistence import ConsolidationStore

# ── UI Configuration ──────────────────────────────────────────────────────────

st.set_page_config(page_title="AdverseScore Clinical AI", page_icon="⚕️", layout='wide')

# ── Design System ─────────────────────────────────────────────────────────────

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

# ── Branded Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div class="logo"><span class="logo-accent">Adverse</span>Score</div>
    <span class="badge badge-neutral">v1.0</span>
    <div class="tagline">PSUR Consolidation — Clinical Decision Support</div>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "in_session_dataset" not in st.session_state:
    st.session_state.in_session_dataset = None
if "consolidation_id" not in st.session_state:
    st.session_state.consolidation_id = None
if "generated_doc_bytes" not in st.session_state:
    st.session_state.generated_doc_bytes = None

# Fresh (cheap) SQLite connection every rerun — Streamlit reruns the whole
# script on every interaction, and caching this in session_state risks
# cross-rerun/thread issues with a long-lived sqlite3 connection.
try:
    store = ConsolidationStore()
except Exception as e:
    st.error(f"Could not open the local consolidation database: {e}")
    st.stop()

PERIOD_OPTIONS = ["6mo", "1yr", "2yr", "3yr"]

RESOLUTION_CONFIDENCE_BADGE = {
    "EXACT": "badge-success",
    "FUZZY": "badge-warning",
    "PARTIAL": "badge-warning",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### New Consolidation")
    drug_name_input = st.text_input("Drug name", key="drug_name_input")
    period_input = st.selectbox("Reporting period", PERIOD_OPTIONS, key="period_input")

    if st.button("Run Consolidation", use_container_width=True):
        if not drug_name_input or not drug_name_input.strip():
            st.warning("Enter a drug name before running a consolidation.")
        else:
            try:
                with st.spinner(f"Consolidating {drug_name_input.strip()} ({period_input})..."):
                    result = consolidate_psur(drug_name_input.strip(), period_input)
            except Exception as e:
                st.error(f"Unexpected error: {e}")
            else:
                if isinstance(result, ConsolidationResult):
                    st.session_state.in_session_dataset = result
                    st.session_state.generated_doc_bytes = None
                    st.session_state.consolidation_id = None
                    try:
                        cid = store.save_consolidation(result)
                    except Exception as e:
                        st.error(f"Consolidation succeeded but saving it failed: {e}")
                    else:
                        st.session_state.consolidation_id = cid
                        st.rerun()
                elif isinstance(result, ConsolidationError):
                    if result.stage == "CLIENT_CONSTRUCTION":
                        st.error(
                            f"Configuration issue: {result.message} — "
                            "check that API keys are set."
                        )
                    elif result.stage == "IDENTITY_RESOLUTION":
                        st.error(
                            f"Could not resolve '{drug_name_input.strip()}': {result.message} — "
                            "check the spelling or try a different name."
                        )
                    elif result.stage == "RETRIEVAL":
                        st.error(
                            f"Data retrieval failed: {result.message} — you can try again."
                        )
                    else:
                        st.error(f"Consolidation failed ({result.stage}): {result.message}")
                else:
                    st.error("Unexpected response from consolidation pipeline.")

    st.markdown("---")
    st.markdown("### Prior Sessions")
    try:
        prior_sessions = store.list_consolidations()
    except Exception as e:
        prior_sessions = []
        st.error(f"Could not load prior sessions: {e}")

    if prior_sessions:
        labels = [
            f"{entry['canonical_name']} — {entry['period']} — {entry['created_at']}"
            for entry in prior_sessions
        ]
        selected_idx = st.selectbox(
            "Select a prior consolidation",
            range(len(prior_sessions)),
            format_func=lambda i: labels[i],
            key="prior_session_idx",
        )
        if st.button("Load Selected", use_container_width=True):
            selected_id = prior_sessions[selected_idx]["id"]
            try:
                loaded = store.load_consolidation(selected_id)
            except Exception as e:
                st.error(f"Unexpected error loading consolidation: {e}")
            else:
                if loaded is not None:
                    st.session_state.in_session_dataset = loaded
                    st.session_state.consolidation_id = selected_id
                    st.session_state.generated_doc_bytes = None
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"Loaded prior consolidation for "
                            f"{loaded.drug_identity.canonical_name} ({loaded.period})."
                        ),
                    })
                    st.rerun()
                else:
                    st.error("Could not find that consolidation — it may have been removed.")
    else:
        st.caption("No prior consolidations saved yet.")


# ── Main Area ──────────────────────────────────────────────────────────────────
dataset = st.session_state.in_session_dataset

if dataset is not None:
    identity = dataset.drug_identity

    # ── Identity Resolution Feedback Card ──────────────────────────────────
    confidence_badge = RESOLUTION_CONFIDENCE_BADGE.get(
        identity.resolution_confidence, "badge-neutral"
    )
    mad_display = (
        identity.market_authorization_date.isoformat()
        if identity.market_authorization_date
        else "OTC — no market authorization date on file"
    )
    st.markdown(f"""
    <div class="card card-accent-primary">
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="font-weight:700; font-size:var(--text-lg);">{html.escape(identity.canonical_name)}</span>
            <span class="badge {confidence_badge}">{html.escape(identity.resolution_confidence)}</span>
        </div>
        <div style="color:var(--color-text-muted); font-size:var(--text-sm); line-height:1.6;">
            <strong>Brand names:</strong> {html.escape(", ".join(identity.brand_names)) or "—"}<br/>
            <strong>Generic names:</strong> {html.escape(", ".join(identity.generic_names)) or "—"}<br/>
            <strong>Market authorization date:</strong> {html.escape(mad_display)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data Completeness Card (Guardrail 5) ───────────────────────────────
    retrieval = dataset.retrieval
    dedup = dataset.dedup
    truncated_chunks = [c.label for c in retrieval.chunks if c.truncated]

    truncated_html = (
        f"<div style='color:var(--color-danger); font-size:var(--text-sm); margin-top:0.4rem;'>"
        f"<strong>Truncated chunks:</strong> {html.escape(', '.join(truncated_chunks))}</div>"
        if truncated_chunks else
        "<div style='color:var(--color-success); font-size:var(--text-sm); margin-top:0.4rem;'>"
        "No chunks truncated — full period coverage retrieved.</div>"
    )

    st.markdown(f"""
    <div class="card">
        <div style="font-weight:700; margin-bottom:0.5rem;">Data Completeness &amp; Methodology</div>
        <div style="font-size:var(--text-sm); line-height:1.7;">
            <strong>Reports retrieved:</strong> {retrieval.total_retrieved_count} of an estimated
            {retrieval.total_estimated_count} total<br/>
            <strong>Period:</strong> {retrieval.period_start} to {retrieval.period_end}
            ({"fallback anchor: " + (retrieval.fallback_reason or "unknown") if retrieval.used_fallback_anchor else "anchored to market authorization date"})<br/>
            <strong>Deduplication:</strong> {dedup.total_input_count} input reports →
            {dedup.total_output_count} unique
            (removed {dedup.removed_by_exact_id} by exact ID match,
            {dedup.removed_by_heuristic} by heuristic match)
        </div>
        {truncated_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Ranked Signal Table ─────────────────────────────────────────────────
    st.markdown("#### Ranked Signals")
    top_signals = dataset.ranking.ranked_signals[:TOP_N_NARRATED_SIGNALS]
    table_rows = [
        {
            "Rank": s.rank,
            "Symptom": s.symptom,
            "Seriousness": s.seriousness_tier,
            "Strength of Evidence": s.strength_of_evidence_tier,
            "Reversibility": s.reversibility_tier,
            "Public Health Impact": s.public_health_tier,
            "Report Count": s.report_count,
        }
        for s in top_signals
    ]
    if table_rows:
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No ranked signals available for this consolidation.")
    st.caption(
        f"Showing top {len(table_rows)} of {dataset.ranking.total_signals} signals — "
        "full ranked list is included in the exported PSUR document."
    )

    # ── Document Generation ─────────────────────────────────────────────────
    st.markdown("#### Export")
    if st.button("Generate PSUR Document"):
        try:
            with st.spinner("Generating document..."):
                doc = generate_psur_document(dataset)
                buf = io.BytesIO()
                doc.save(buf)
                buf.seek(0)
                st.session_state.generated_doc_bytes = buf.getvalue()
        except Exception as e:
            st.error(f"Unexpected error generating document: {e}")

    if st.session_state.generated_doc_bytes is not None:
        st.download_button(
            "Download .docx",
            data=st.session_state.generated_doc_bytes,
            file_name=f"PSUR_{identity.canonical_name}_{dataset.period}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    st.markdown("---")
else:
    st.info(
        "Run a consolidation from the sidebar (or load a prior session) to view "
        "identity resolution, data completeness, and ranked signals."
    )

# ── Chat History Display ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Interface ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about this drug's safety profile..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history = st.session_state.messages[:-1]
        try:
            with st.spinner("Thinking..."):
                turn_result = run_agent_turn(
                    conversation_history=history,
                    user_message=prompt,
                    in_session_dataset=st.session_state.in_session_dataset,
                )
        except Exception as e:
            error_text = f"Unexpected error: {e}"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})
        else:
            st.markdown(turn_result.response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": turn_result.response_text}
            )

            try:
                store.save_message(
                    st.session_state.session_id, "user", prompt, st.session_state.consolidation_id
                )
                store.save_message(
                    st.session_state.session_id,
                    "assistant",
                    turn_result.response_text,
                    st.session_state.consolidation_id,
                )
            except Exception as e:
                st.error(f"Response generated, but saving chat history failed: {e}")

            if turn_result.dataset_changed:
                st.session_state.in_session_dataset = turn_result.updated_dataset
                st.session_state.generated_doc_bytes = None
                try:
                    new_cid = store.save_consolidation(turn_result.updated_dataset)
                except Exception as e:
                    st.error(f"New dataset generated, but saving it failed: {e}")
                else:
                    st.session_state.consolidation_id = new_cid
                    st.rerun()
