"""Agent tool boundary — Phase 8 of the PSUR consolidation rebuild.

The old single `get_adverse_score` tool tied to the five-capability composite
score was removed as part of the PSUR consolidation rebuild (Phase 0, see
docs/PSUR_CONSOLIDATION_SCOPE.md). This module wraps the two Phase 1/6 entry
points (`drug_identity.resolve_drug_identity`, `consolidation.consolidate_psur`)
as LangChain `@tool`-decorated callables for `orchestrator.py`'s LangGraph
agent.

Both tools return plain, JSON-serializable dicts — the shape an LLM tool-call
result gets serialized into and fed back into the message history. Frozen
dataclasses and `date` objects are never returned directly (LangGraph would
choke serializing them into a ToolMessage); this module is exactly where that
translation happens, matching the boundary `drug_identity.py`'s own docstring
calls out ("this module does NOT use the agent-facing payload shape ... that
translation belongs to Phase 8's agent_tools.py").

No HTTP calls are made directly in this module — both tools delegate to
already-HTTP-capable Phase 1/6 modules.
"""
import contextvars
import dataclasses
from typing import Optional

from langchain_core.tools import tool

from .config import TOP_N_NARRATED_SIGNALS
from .consolidation import consolidate_psur, ConsolidationError, ConsolidationResult
from .drug_identity import resolve_drug_identity, DrugIdentityError
from .logger import get_logger, log_event

logger = get_logger("agent_tools")


# ── Out-of-band full-result capture ────────────────────────────────────────
# `consolidate_psur_tool`'s LLM-visible return value is deliberately a bounded
# summary dict (see its docstring below) — a full `ConsolidationResult`, with
# its thousands of nested per-report dicts, would blow the LLM's context
# window if it flowed back through the tool-message channel. But
# `orchestrator.py`'s `run_agent_turn()` needs the FULL `ConsolidationResult`
# object (dates, nested dataclasses and all) to populate
# `AgentTurnResult.updated_dataset`, and LangGraph only threads a tool's
# *return value* back into the message history — there is no built-in side
# channel for "also hand the caller this other, bigger object."
#
# Design chosen: a `contextvars.ContextVar` holding a mutable list ("capture
# box"), NOT a bare module-level variable and NOT `threading.local()`.
# `threading.local()` was tried first and does not work here: LangGraph's
# prebuilt `ToolNode` dispatches synchronous tool calls onto an internal
# worker thread (confirmed empirically during Phase 8 build — a tool prints
# `threading.current_thread()` and it is NOT the thread that called
# `graph.invoke()`), so thread-local storage written inside the tool is
# invisible to `orchestrator.py`'s calling thread.
#
# `contextvars.ContextVar` values DO cross this same thread hop (Python's
# executor bridge copies the calling context into the worker thread), but
# only for *mutation of an already-referenced object* — rebinding the
# ContextVar itself from within a copied context (a `.set()` call made
# inside the tool) does NOT propagate back to the caller's context. This
# holds even when there is no literal thread hop: LangChain's `Runnable.
# invoke()` already runs the callable inside a `contextvars.copy_context()`
# for callback/config isolation, so a tool can never reach back out and
# rebind a ContextVar in its caller's context, whether or not a real OS
# thread boundary is also involved (verified experimentally during Phase 8
# build, both with and without going through a LangGraph tool node).
#
# So the pattern here is REQUIRED, not optional: the caller
# (`orchestrator.run_agent_turn`, via `begin_consolidation_capture()`)
# pre-creates an empty list and `.set()`s the ContextVar to reference it
# BEFORE calling `graph.invoke()` (or, for direct/standalone tool
# invocation, before calling `.invoke()`). The tool, running in a copied
# context (possibly on a different thread), calls `.get()` to reach that
# SAME list object and mutates it in place (`.append()`) — a mutation, not a
# rebind, so it stays visible in the list object the caller is still holding
# a reference to. If `begin_consolidation_capture()` was never called, the
# capture box is `None` and `_stash_consolidation_result` is a documented
# no-op — there is nothing it could write into that would be visible to any
# caller.
_capture_var: "contextvars.ContextVar" = contextvars.ContextVar(
    "adversescore_consolidation_capture", default=None
)


def begin_consolidation_capture() -> None:
    """Installs a fresh, empty capture box in the CURRENT context. Callers
    MUST call this once per turn/invocation, before invoking the agent graph
    (or calling `consolidate_psur_tool` directly), so the tool's `.append()`
    (see module docstring above) has something to write into that's still
    reachable by the caller afterwards. Without this call first,
    `consolidate_psur_tool` still runs and returns its summary dict
    normally — only the out-of-band full-result capture is skipped."""
    _capture_var.set([])


def _stash_consolidation_result(result: ConsolidationResult) -> None:
    box = _capture_var.get()
    if box is None:
        # No capture box was pre-installed via begin_consolidation_capture().
        # There is nothing reachable to write into (see module docstring —
        # a `.set()` here would only rebind this copied context's view of
        # the ContextVar, invisible to the caller) — documented no-op.
        return
    box.append(result)


def pop_last_consolidation_result() -> Optional[ConsolidationResult]:
    """Reads and clears the most recently captured full `ConsolidationResult`
    (stashed by `consolidate_psur_tool` on its most recent successful call in
    this context's capture box). Returns `None` if the tool was not called,
    was called and failed, or no capture box exists yet."""
    box = _capture_var.get()
    if not box:
        return None
    result = box[-1]
    box.clear()
    return result


# ── Tools ────────────────────────────────────────────────────────────────
# Both tools are the LLM's only bridge into Phase 1-6 pipeline code. LangGraph's
# `create_agent` (the installed langchain==1.2.11's API — see orchestrator.py's
# module docstring) does not expose a way to wire a custom `ToolNode(...,
# handle_tool_errors=True)` through it, so by default ANY exception escaping a
# tool function crashes the entire graph run (verified empirically during
# Phase 8 build: an LLM passing a malformed argument, e.g. period="last 6
# months" instead of the required literal "6mo", raised a raw KeyError deep in
# fda_client.py that propagated uncaught all the way out of graph.invoke()).
# Both tools below are therefore defensive at two layers: (1) validate
# arguments the pipeline modules assume are already well-formed before calling
# them, and (2) wrap the pipeline call in a broad `except Exception` so no
# malformed LLM tool-call argument or unexpected pipeline bug can ever crash a
# whole agent turn — worst case, the LLM (or the end user) sees a structured
# error dict instead.

_VALID_PERIODS = ("6mo", "1yr", "2yr", "3yr")


@tool
def resolve_drug_identity_tool(drug_name: str) -> dict:
    """Look up a drug's canonical identity (brand/generic names, market
    authorization date) without running a full PSUR consolidation. Use for
    quick identity questions ('is X the same as Y?') that don't need the
    full signal dataset."""
    try:
        result = resolve_drug_identity(drug_name)
    except Exception as e:  # pipeline bug or malformed input — never crash the turn
        log_event(logger, "agent_tool_resolve_identity_unexpected_exception",
                  drug_name=drug_name, error=str(e))
        return {"error": "UNEXPECTED_EXCEPTION", "message": str(e)}

    if isinstance(result, DrugIdentityError):
        log_event(logger, "agent_tool_resolve_identity_error",
                  drug_name=drug_name, reason=result.reason)
        return {"error": result.reason, "message": result.message}

    payload = dataclasses.asdict(result)
    mad = payload.get("market_authorization_date")
    # dates aren't JSON/LLM-serializable as-is — same .isoformat()-if-not-None
    # pattern persistence.py already established for this exact field.
    payload["market_authorization_date"] = mad.isoformat() if mad else None

    log_event(logger, "agent_tool_resolve_identity_success",
              drug_name=drug_name, canonical_name=payload.get("canonical_name"))
    return payload


@tool
def consolidate_psur_tool(drug_name: str, period: str) -> dict:
    """Run the full PSUR consolidation pipeline for a drug + period. Returns
    ranked signals, completeness metadata, dedup stats, and label status
    breakdown. This is the primary data-gathering tool — expensive (can take
    minutes for high-volume drugs), so only call it when the user is asking
    about a NEW drug/period not already in the current session's dataset.

    `period` MUST be exactly one of the four literal strings "6mo", "1yr",
    "2yr", or "3yr" — no other phrasing (e.g. "last 6 months", "one year")
    is accepted. Translate the user's natural-language period into one of
    these four values before calling this tool."""
    if period not in _VALID_PERIODS:
        log_event(logger, "agent_tool_consolidate_invalid_period",
                  drug_name=drug_name, period=period)
        return {
            "error": "INVALID_PERIOD",
            "message": (
                f"period must be exactly one of {_VALID_PERIODS!r}, got "
                f"{period!r}. Translate the user's requested timeframe into "
                "one of these four literal values and try again."
            ),
        }

    try:
        result = consolidate_psur(drug_name, period)
    except Exception as e:  # pipeline bug or malformed input — never crash the turn
        log_event(logger, "agent_tool_consolidate_unexpected_exception",
                  drug_name=drug_name, period=period, error=str(e))
        return {"error": "UNEXPECTED_EXCEPTION", "message": str(e)}

    if isinstance(result, ConsolidationError):
        log_event(logger, "agent_tool_consolidate_error", drug_name=drug_name,
                  period=period, stage=result.stage, reason=result.reason)
        return {"error": result.stage, "reason": result.reason, "message": result.message}

    # Full result captured out-of-band for orchestrator.py — see module
    # docstring above. The dict returned below is what the LLM sees; it is
    # NOT sufficient on its own to reconstruct `result` (no report-level
    # data, no drug_identity/retrieval/dedup nested dataclasses).
    _stash_consolidation_result(result)

    top_signals = [
        {
            "symptom": signal.symptom,
            "rank": signal.rank,
            "seriousness_tier": signal.seriousness_tier,
            "strength_of_evidence_tier": signal.strength_of_evidence_tier,
            "reversibility_tier": signal.reversibility_tier,
            "public_health_tier": signal.public_health_tier,
            "report_count": signal.report_count,
            "prr_metrics": signal.prr_metrics,
        }
        for signal in result.ranking.ranked_signals[:TOP_N_NARRATED_SIGNALS]
    ]

    summary = {
        "canonical_name": result.drug_identity.canonical_name,
        "period": result.period,
        "total_retrieved_count": result.retrieval.total_retrieved_count,
        "total_estimated_count": result.retrieval.total_estimated_count,
        "any_chunk_truncated": result.retrieval.any_chunk_truncated,
        "dedup_total_input_count": result.dedup.total_input_count,
        "dedup_total_output_count": result.dedup.total_output_count,
        "dedup_removed_by_exact_id": result.dedup.removed_by_exact_id,
        "dedup_removed_by_heuristic": result.dedup.removed_by_heuristic,
        "formula_version": result.formula_version,
        "total_signals": result.ranking.total_signals,
        "top_signals": top_signals,
    }

    log_event(logger, "agent_tool_consolidate_success", drug_name=drug_name,
              period=period, total_signals=result.ranking.total_signals)
    return summary
