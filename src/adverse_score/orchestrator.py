"""Agent orchestration — Phase 8 of the PSUR consolidation rebuild.

The old single GPT-4o react-agent wired to the five-capability composite-score
tool was removed as part of the PSUR consolidation rebuild (Phase 0, see
docs/PSUR_CONSOLIDATION_SCOPE.md). This module rebuilds it around the Phase
1-6 pipeline: a LangGraph tool-calling agent (`langchain.agents.create_agent`,
which compiles to a LangGraph `CompiledStateGraph` — this repo's committed
agent-graph framework) wired to the two `agent_tools.py` tools, plus a system
prompt and a set of deterministic, code-enforced post-processing guardrails.

Import-safety convention (matches `fda_client.py`'s `FDAClient.__init__` and
this module's own prior placeholder docstring): importing this module must
never require `OPENAI_API_KEY`/`OPENFDA_API_KEY`. Only *calling*
`run_agent_turn()` constructs the `ChatOpenAI` model (which validates
credentials at construction time) and only on first call — the compiled
graph is then cached at module level for subsequent calls.

Persistence is deliberately NOT this module's job: `run_agent_turn()` takes
`conversation_history` and `in_session_dataset` as plain arguments and
returns a plain `AgentTurnResult` — no `ConsolidationStore` or `persistence`
import anywhere in this file. The caller (eventually app.py, Phase 10) owns
loading/saving both.
"""
from dataclasses import dataclass
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from .agent_tools import (
    begin_consolidation_capture,
    consolidate_psur_tool,
    pop_last_consolidation_result,
    resolve_drug_identity_tool,
)
from .config import (
    AGENT_MAX_ITERATIONS,
    AGENT_TEMPERATURE,
    CONVERSATION_ROLE_ASSISTANT,
    CONVERSATION_ROLE_SYSTEM,
    CONVERSATION_ROLE_USER,
    OPENAI_CHAT_MODEL,
    TOP_N_NARRATED_SIGNALS,
)
from .consolidation import ConsolidationResult
from .logger import get_logger, log_event

logger = get_logger("orchestrator")


# ── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the AdverseScore PSUR consolidation assistant, supporting \
pharmacovigilance (PV) professionals in preparing Periodic Safety Update Reports \
(PSURs) aligned to ICH E2C(R2) PBRER format.

SCOPE (Guardrail 7 - Scope Enforcement): Your job is PSUR consolidation only - \
retrieving, deduplicating, and ranking FAERS adverse event signals for a specific \
drug and reporting period. You do NOT provide general medical advice, prescribing \
guidance, dosing recommendations, or answer unrelated drug questions. If asked \
something outside this scope (for example "should I take X for my headache" or \
"what dose of Z should I use"), politely decline to answer the underlying medical \
question and redirect the user back to PSUR consolidation.

CAUSALITY LOCK (Guardrail 1): Never assert or imply that a drug CAUSES an adverse \
event. Signal ranking reflects statistical association (PRR, reporting volume, \
seriousness, reversibility) only - never causation. Prefer language like \
"associated with," "reported alongside," or "a signal was detected for" over \
"causes," "results in," or "led to." Never recommend medication changes, dosing \
adjustments, discontinuation, or any patient-level clinical action - that is a \
qualified reviewer's job, not yours.

DATASET REUSE: If a dataset is already available in this session for the drug and \
period being discussed (see the session context message, if present), answer from \
it rather than calling consolidate_psur_tool again. consolidate_psur_tool is \
expensive (it can take minutes for high-volume drugs) - only call it for a \
genuinely NEW drug/period not already in the current session's dataset. Use \
resolve_drug_identity_tool instead for quick identity-only questions that don't \
need the full signal dataset.

PERIOD FORMAT: consolidate_psur_tool's period argument must be exactly one of the \
four literal strings "6mo", "1yr", "2yr", or "3yr" - never a natural-language \
phrase like "the last six months" or "one year". Always translate the user's \
requested timeframe into one of these four literal values before calling the \
tool; if their requested timeframe doesn't map cleanly onto one of the four, ask \
them to pick one of "6mo", "1yr", "2yr", or "3yr" instead of guessing.

NARRATION CAP: When discussing ranked signals in conversation, limit detail to at \
most the top {top_n} signals. The full ranked list is not exportable yet in this \
phase of the product - do not reference document, PDF, or DOCX export \
capabilities.

Always be precise about what the data does and does not show, and never overstate \
certainty.""".format(top_n=TOP_N_NARRATED_SIGNALS)


# ── Output Data Model ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentTurnResult:
    response_text: str
    updated_dataset: Optional[ConsolidationResult]   # current in-session dataset after this turn
    dataset_changed: bool                             # True if a new consolidation happened this turn


# ── Guardrail Enforcement (deterministic, code-side half) ─────────────────
# Guardrails 2/3 (no-fabrication placeholder, draft marking) are explicitly
# NOT this phase's job - Phase 9 (document generation) owns those.

def _flag_causal_language(text: str) -> bool:
    """Guardrail 1 (Causality Lock), code-enforced half. Scans `text` for
    causal-assertion phrasing (e.g. "causes", "results in", "led to the").
    Does NOT block or rewrite the response - the primary enforcement
    mechanism is the system prompt instruction above; this is a logging-only
    safety net so causal-language slips are auditable. Returns True if any
    causal pattern was found (and, as a side effect, logs a warning via
    `log_event`)."""
    lowered = text.lower()
    causal_patterns = (
        "causes", "cause of", "is known to cause", "caused by",
        "results in", "resulted in", "results from",
        "led to the", "leads to the", "due to the drug",
    )
    flagged = any(pattern in lowered for pattern in causal_patterns)
    if flagged:
        log_event(logger, "causal_language_flagged", snippet=text[:200])
    return flagged


def _ensure_human_review_reminder(text: str, dataset_referenced: bool) -> str:
    """Guardrail 4 (Human Review Requirement), code-enforced. If a dataset is
    in play this turn (`dataset_referenced=True`) and `text` doesn't already
    contain a review-reminder phrase, appends a fixed reminder sentence.
    Idempotent - never duplicates an existing reminder."""
    if not dataset_referenced:
        return text

    lowered = text.lower()
    review_markers = ("human review", "clinical review", "pending review", "regulatory review")
    if any(marker in lowered for marker in review_markers):
        return text

    return text + "\n\n*This analysis requires qualified clinical/regulatory review before any use.*"


def _ensure_completeness_statement(text: str, dataset: Optional[ConsolidationResult]) -> str:
    """Guardrail 5 (Completeness/Methodology Transparency) AND Guardrail 6
    (Formula Version Audit Trail), both code-enforced, folded into one
    appended paragraph (avoids stacking multiple methodology notes for what
    is conceptually a single disclosure). If `dataset` is present and `text`
    doesn't already reference BOTH retrieval-completeness data AND the
    formula version, appends a line built from
    `dataset.retrieval.total_retrieved_count`,
    `dataset.retrieval.total_estimated_count`,
    `dataset.retrieval.any_chunk_truncated`, and `dataset.formula_version`.
    Idempotent - never duplicates an existing statement.

    Deliberately requires BOTH pieces of evidence (not just one) before
    skipping the append: guardrail 5 (completeness) and guardrail 6
    (formula version audit trail) are two distinct disclosure requirements
    folded into one paragraph for brevity, but that folding must not let
    either one be silently dropped. A response that merely uses the word
    "retrieved" in passing (e.g. "5 signals were retrieved and ranked...")
    must NOT suppress the formula-version audit-trail disclosure it never
    actually made - checking a single generic marker like "retrieved" in
    isolation previously allowed exactly that silent suppression."""
    if dataset is None:
        return text

    lowered = text.lower()
    has_completeness_info = any(marker in lowered for marker in ("retrieved", "completeness"))
    has_formula_version = "formula version" in lowered or "formula_version" in lowered
    if has_completeness_info and has_formula_version:
        return text

    retrieval = dataset.retrieval
    truncation_note = (
        "some data chunks were truncated by openFDA pagination limits"
        if retrieval.any_chunk_truncated
        else "no chunk truncation occurred"
    )
    completeness_line = (
        f"\n\n*Methodology: {retrieval.total_retrieved_count} of an estimated "
        f"{retrieval.total_estimated_count} matching FAERS reports were retrieved "
        f"({truncation_note}); ranking formula version {dataset.formula_version}.*"
    )
    return text + completeness_line


def _apply_guardrail_postprocessing(
    text: str, dataset: Optional[ConsolidationResult], dataset_changed: bool
) -> str:
    """Combines the code-enforced guardrail post-processing steps into one
    pass over a turn's response text. `dataset` is the dataset in play after
    this turn (freshly consolidated or carried forward) - `None` if no
    dataset has ever been established in this session."""
    causal_flagged = _flag_causal_language(text)
    dataset_referenced = dataset is not None

    text = _ensure_human_review_reminder(text, dataset_referenced)
    text = _ensure_completeness_statement(text, dataset)

    log_event(
        logger, "guardrail_postprocessing_applied",
        dataset_changed=dataset_changed, dataset_referenced=dataset_referenced,
        causal_language_flagged=causal_flagged,
    )
    return text


# ── Agent Graph Construction (lazy, cached on first call) ─────────────────
# `_cached_graph` is a module-level cache populated on first `run_agent_turn()`
# call, NOT at import time - see module docstring for the import-safety
# rationale. Tests substitute a fake-model-backed graph by monkeypatching
# this module attribute directly (`monkeypatch.setattr(orchestrator,
# "_cached_graph", fake_graph)`), bypassing `ChatOpenAI` construction
# entirely without needing any extra public API surface.
_cached_graph = None


def _get_agent_graph():
    global _cached_graph
    if _cached_graph is None:
        model = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=AGENT_TEMPERATURE)
        _cached_graph = create_agent(
            model,
            tools=[resolve_drug_identity_tool, consolidate_psur_tool],
            system_prompt=SYSTEM_PROMPT,
        )
    return _cached_graph


# Empirically derived against the installed langgraph==1.1.0 / langchain==1.2.11
# `create_agent` graph shape: each tool-calling iteration (agent node -> tool
# node) costs 2 graph super-steps, plus 2 more super-steps to enter the graph
# and produce the final no-tool-call response. AGENT_MAX_ITERATIONS therefore
# maps to a `recursion_limit` of `2 * AGENT_MAX_ITERATIONS + 2` - verified by
# scripting a fake LLM that always requests a tool call and confirming the
# graph raises langgraph.errors.GraphRecursionError exactly at this boundary
# (see tests/unit/test_orchestrator.py's iteration-cap test).
_RECURSION_LIMIT = 2 * AGENT_MAX_ITERATIONS + 2


# ── Message Translation ────────────────────────────────────────────────────

def _dataset_context_message(dataset: Optional[ConsolidationResult]) -> Optional[SystemMessage]:
    """A compact session-context note (canonical_name/period/total_signals/
    formula_version only - never the full nested object) telling the LLM a
    dataset is already available, per the system prompt's DATASET REUSE
    instruction. Returns None if no dataset is in session."""
    if dataset is None:
        return None
    return SystemMessage(content=(
        "Session context: a consolidated PSUR dataset is already available - "
        f"canonical_name={dataset.drug_identity.canonical_name}, period={dataset.period}, "
        f"total_signals={dataset.ranking.total_signals}, "
        f"formula_version={dataset.formula_version}. If the user's next question "
        "concerns this same drug and period, answer from this dataset rather than "
        "calling consolidate_psur_tool again."
    ))


def _to_lc_messages(conversation_history: list, user_message: str,
                     dataset: Optional[ConsolidationResult]) -> list:
    """Translates the caller's plain-dict conversation history (plus the new
    user message and current dataset context) into LangChain message objects
    for the graph's `messages` state key."""
    messages = []

    context_message = _dataset_context_message(dataset)
    if context_message is not None:
        messages.append(context_message)

    for turn in conversation_history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == CONVERSATION_ROLE_USER:
            messages.append(HumanMessage(content=content))
        elif role == CONVERSATION_ROLE_ASSISTANT:
            messages.append(AIMessage(content=content))
        elif role == CONVERSATION_ROLE_SYSTEM:
            messages.append(SystemMessage(content=content))
        # Unrecognized roles are silently skipped - malformed history should
        # not crash a turn.

    messages.append(HumanMessage(content=user_message))
    return messages


def _extract_response_text(state: dict) -> str:
    """Pulls the final assistant-visible text out of the graph's terminal
    state. Defensively handles both plain-string AIMessage content and the
    list-of-content-block shape some providers use."""
    messages = state.get("messages") or []
    if not messages:
        return ""

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return ""

    content = last_message.content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content or ""


# ── Entry Point ────────────────────────────────────────────────────────────

def run_agent_turn(
    conversation_history: list,
    user_message: str,
    in_session_dataset: Optional[ConsolidationResult] = None,
) -> AgentTurnResult:
    """Runs one conversational turn of the PSUR consolidation agent.

    Builds (or reuses the cached) LangGraph tool-calling agent, runs it with
    `conversation_history + [user_message]` (plus a compact in-session
    dataset context message, if any), and returns the assistant's response
    together with the current in-session dataset state.

    `updated_dataset`/`dataset_changed`: if `consolidate_psur_tool` ran and
    succeeded this turn, the full `ConsolidationResult` it captured (see
    `agent_tools.py`'s contextvars-based out-of-band capture mechanism)
    becomes `updated_dataset` and `dataset_changed=True`. Otherwise
    `in_session_dataset` is carried forward unchanged and
    `dataset_changed=False`.

    Never persists anything - no `ConsolidationStore`/persistence dependency.
    Requires `OPENAI_API_KEY` (and, transitively via the tools,
    `OPENFDA_API_KEY`) only at call time, not at import time.
    """
    graph = _get_agent_graph()
    messages = _to_lc_messages(conversation_history, user_message, in_session_dataset)

    # Pre-install this turn's capture box in the CURRENT context before the
    # graph runs — see agent_tools.py's module docstring for why this must
    # happen here, before graph.invoke(), rather than lazily inside the tool.
    begin_consolidation_capture()

    hit_iteration_cap = False
    try:
        state = graph.invoke({"messages": messages}, config={"recursion_limit": _RECURSION_LIMIT})
        response_text = _extract_response_text(state)
    except GraphRecursionError:
        hit_iteration_cap = True
        log_event(logger, "agent_turn_iteration_cap_hit", max_iterations=AGENT_MAX_ITERATIONS)
        response_text = (
            "I wasn't able to finish this request within the allotted number of tool "
            "calls. Please try narrowing your question (for example, a single drug "
            "and reporting period) and ask again."
        )

    captured_result = pop_last_consolidation_result()
    if captured_result is not None:
        updated_dataset = captured_result
        dataset_changed = True
    else:
        updated_dataset = in_session_dataset
        dataset_changed = False

    response_text = _apply_guardrail_postprocessing(response_text, updated_dataset, dataset_changed)

    log_event(
        logger, "agent_turn_complete",
        dataset_changed=dataset_changed, hit_iteration_cap=hit_iteration_cap,
    )

    return AgentTurnResult(
        response_text=response_text,
        updated_dataset=updated_dataset,
        dataset_changed=dataset_changed,
    )
