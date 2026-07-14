"""
Unit tests for orchestrator.py (Phase 8) — LangGraph tool-calling agent,
deterministic guardrail post-processing, and AgentTurnResult wiring.

The LLM is never called live: a `langchain_core.language_models.fake_chat_models.
FakeMessagesListChatModel` subclass (overriding `bind_tools` to be a no-op passthrough,
since the fake model doesn't natively support tool binding) scripts exact
AIMessage/tool-call sequences. This establishes the mocking pattern for this repo's
LangGraph-based agent tests going forward.
"""
import re
from datetime import date
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from adverse_score import orchestrator
from adverse_score.consolidation import ConsolidationError, ConsolidationResult
from adverse_score.deduplication import DedupResult
from adverse_score.drug_identity import DrugIdentity, DrugIdentityError
from adverse_score.fda_client import ChunkResult, PSURRetrievalResult
from adverse_score.label_classifier import LabelClassificationResult
from adverse_score.orchestrator import (
    AgentTurnResult,
    _apply_guardrail_postprocessing,
    _ensure_completeness_statement,
    _ensure_human_review_reminder,
    _flag_causal_language,
    run_agent_turn,
)
from adverse_score.ranking import RankedSignal, RankingResult


# ── Fake LLM plumbing ───────────────────────────────────────────────────────

class BindableFakeChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel does not implement bind_tools (raises
    NotImplementedError) — langgraph's create_agent calls model.bind_tools(tools)
    unconditionally at graph-build time. Overriding it as a no-op passthrough is
    the standard workaround for scripting tool-calling behavior with a fake LLM."""

    def bind_tools(self, tools, **kwargs: Any):
        return self


def _build_graph(responses):
    from langchain.agents import create_agent
    from adverse_score.agent_tools import consolidate_psur_tool, resolve_drug_identity_tool

    model = BindableFakeChatModel(responses=responses)
    return create_agent(
        model,
        tools=[resolve_drug_identity_tool, consolidate_psur_tool],
        system_prompt=orchestrator.SYSTEM_PROMPT,
    )


@pytest.fixture
def fake_graph(monkeypatch):
    """Injects a fake-LLM-backed graph via the module-level cache, bypassing
    ChatOpenAI construction entirely. Returns a setter the test calls with its
    scripted response list."""
    def _install(responses):
        graph = _build_graph(responses)
        monkeypatch.setattr(orchestrator, "_cached_graph", graph)
        return graph
    return _install


# ── Fixtures / builders ─────────────────────────────────────────────────────

def _drug_identity(**overrides):
    defaults = dict(
        canonical_name="KEYTRUDA",
        brand_names=["KEYTRUDA"],
        generic_names=["PEMBROLIZUMAB"],
        substance_names=["PEMBROLIZUMAB"],
        application_numbers=["BLA125514"],
        market_authorization_date=date(2014, 9, 4),
        market_authorization_date_source="drugsfda.json:BLA125514",
        resolution_confidence="EXACT",
    )
    defaults.update(overrides)
    return DrugIdentity(**defaults)


def _consolidation_result(n_signals=3, any_chunk_truncated=False, **overrides):
    identity = _drug_identity()
    chunk = ChunkResult(
        label="chunk-0-20240101-to-20240401", start_date="20240101", end_date="20240401",
        reports=[], retrieved_count=100, estimated_total_count=100,
        truncated=any_chunk_truncated, error=None,
    )
    retrieval = PSURRetrievalResult(
        canonical_query_variants=["KEYTRUDA"], period_start="20240101", period_end="20240701",
        chunks=[chunk], reports=[], total_retrieved_count=100, total_estimated_count=120,
        any_chunk_truncated=any_chunk_truncated, used_fallback_anchor=False, fallback_reason=None,
    )
    dedup = DedupResult(
        reports=[], total_input_count=100, total_output_count=90,
        removed_by_exact_id=8, removed_by_heuristic=2, audit_trail=[],
    )
    ranked_signals = [
        RankedSignal(
            symptom=f"SIGNAL_{i}", rank=i + 1, seriousness_tier="NON_SERIOUS",
            strength_of_evidence_tier="WEAK_UNLABELED", reversibility_tier="UNKNOWN",
            public_health_tier="LOW",
            prr_metrics={"prr": 1.0, "ci_lower": 0.5, "signal_detected": False,
                         "target_symptom": f"SIGNAL_{i}", "drug_cases": 5,
                         "class_cases": 10, "label_status": "UNLABELED"},
            report_count=5,
        )
        for i in range(n_signals)
    ]
    label_summary = LabelClassificationResult(
        statuses={s.symptom: "UNLABELED" for s in ranked_signals},
        total_symptoms=n_signals, labeled_count=0, unlabeled_count=n_signals, unknown_count=0,
    )
    ranking = RankingResult(
        ranked_signals=ranked_signals, label_summary=label_summary,
        formula_version="1.0", total_signals=n_signals,
    )
    defaults = dict(
        drug_identity=identity, period="6mo", retrieval=retrieval, dedup=dedup,
        ranking=ranking, pharm_class="PD-1 inhibitors", class_counts_available=True,
        formula_version="1.0",
    )
    defaults.update(overrides)
    return ConsolidationResult(**defaults)


def _tool_call_message(tool_name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])


# ── Guardrail: Causality Lock (log-only) ────────────────────────────────────

class TestFlagCausalLanguage:
    def test_flags_obvious_causal_phrases(self):
        assert _flag_causal_language("This drug causes liver damage.") is True
        assert _flag_causal_language("The adverse event was caused by the drug.") is True
        assert _flag_causal_language("This finding led to the conclusion of harm.") is True

    def test_does_not_flag_statistical_language(self):
        assert _flag_causal_language("Nausea is associated with this drug per PRR analysis.") is False
        assert _flag_causal_language("A signal was detected for hepatotoxicity.") is False
        assert _flag_causal_language("This report was reported alongside three others.") is False

    def test_logs_event_when_flagged(self):
        with patch("adverse_score.orchestrator.log_event") as mock_log:
            _flag_causal_language("This drug causes fatigue.")
            mock_log.assert_called_once()
            assert mock_log.call_args.args[1] == "causal_language_flagged"

    def test_no_log_when_not_flagged(self):
        with patch("adverse_score.orchestrator.log_event") as mock_log:
            _flag_causal_language("A signal was detected for nausea.")
            mock_log.assert_not_called()

    def test_never_rewrites_text(self):
        """Guardrail 1's code half only logs — it must never mutate the response."""
        original = "This drug causes severe reactions."
        _flag_causal_language(original)
        assert original == "This drug causes severe reactions."


# ── Guardrail 4: Human Review Requirement ───────────────────────────────────

class TestEnsureHumanReviewReminder:
    def test_appends_when_missing_and_dataset_referenced(self):
        text = "Here is the signal summary."
        result = _ensure_human_review_reminder(text, dataset_referenced=True)
        assert result != text
        assert "review" in result.lower()
        assert result.startswith(text)

    def test_does_not_append_when_not_referenced(self):
        text = "Hello, how can I help?"
        result = _ensure_human_review_reminder(text, dataset_referenced=False)
        assert result == text

    def test_does_not_duplicate_when_already_present(self):
        text = "Here is the summary. This requires human review before use."
        result = _ensure_human_review_reminder(text, dataset_referenced=True)
        assert result == text
        assert result.lower().count("human review") == 1

    def test_case_insensitive_detection(self):
        text = "Please note this needs CLINICAL REVIEW."
        result = _ensure_human_review_reminder(text, dataset_referenced=True)
        assert result == text


# ── Guardrails 5+6: Completeness / Formula Version Audit Trail ─────────────

class TestEnsureCompletenessStatement:
    def test_appends_when_missing_and_dataset_present(self):
        dataset = _consolidation_result()
        text = "Here are the top signals."
        result = _ensure_completeness_statement(text, dataset)
        assert result != text
        assert "100" in result  # total_retrieved_count
        assert "120" in result  # total_estimated_count
        assert "1.0" in result  # formula_version

    def test_no_dataset_returns_text_unchanged(self):
        text = "General information about PSUR consolidation."
        result = _ensure_completeness_statement(text, None)
        assert result == text

    def test_does_not_duplicate_when_already_present(self):
        dataset = _consolidation_result()
        text = "100 reports were retrieved out of an estimated total (formula version 1.0)."
        result = _ensure_completeness_statement(text, dataset)
        assert result == text

    def test_completeness_marker_alone_does_not_suppress_formula_version_disclosure(self):
        """Regression test: a response that merely uses the word "retrieved" in
        passing, without ever disclosing the formula version, must still get
        the completeness statement appended - guardrail 6 (formula version
        audit trail) must never be silently dropped just because guardrail 5's
        marker word happens to appear somewhere in the text."""
        dataset = _consolidation_result()
        text = "5 signals were retrieved and ranked for KEYTRUDA over the last 6 months."
        result = _ensure_completeness_statement(text, dataset)
        assert result != text
        assert "formula version" in result.lower()

    def test_formula_version_marker_alone_does_not_suppress_completeness_disclosure(self):
        """Symmetric regression test: mentioning the formula version alone,
        without any retrieval-completeness figures, must still get the
        completeness statement appended."""
        dataset = _consolidation_result()
        text = "This ranking used formula version 1.0."
        result = _ensure_completeness_statement(text, dataset)
        assert result != text
        assert "100" in result

    def test_mentions_truncation_when_any_chunk_truncated(self):
        dataset = _consolidation_result(any_chunk_truncated=True)
        result = _ensure_completeness_statement("Summary.", dataset)
        assert "truncat" in result.lower()

    def test_sources_numbers_from_real_dataclass_fields(self):
        dataset = _consolidation_result()
        dataset = ConsolidationResult(
            drug_identity=dataset.drug_identity,
            period=dataset.period,
            retrieval=PSURRetrievalResult(
                canonical_query_variants=["FOO"], period_start="20240101", period_end="20240701",
                chunks=[], reports=[], total_retrieved_count=555, total_estimated_count=777,
                any_chunk_truncated=False, used_fallback_anchor=False, fallback_reason=None,
            ),
            dedup=dataset.dedup, ranking=dataset.ranking, pharm_class=dataset.pharm_class,
            class_counts_available=dataset.class_counts_available, formula_version="2.5",
        )
        result = _ensure_completeness_statement("Summary.", dataset)
        assert "555" in result
        assert "777" in result
        assert "2.5" in result


class TestApplyGuardrailPostprocessing:
    def test_combines_both_appends(self):
        dataset = _consolidation_result()
        text = "Here are the results."
        result = _apply_guardrail_postprocessing(text, dataset, dataset_changed=True)
        assert "review" in result.lower()
        assert "100" in result

    def test_no_dataset_no_appends(self):
        text = "Hello, how can I help you today?"
        result = _apply_guardrail_postprocessing(text, None, dataset_changed=False)
        assert result == text


# ── Tool-calling flow via a fake LLM ───────────────────────────────────────

class TestRunAgentTurnToolCalling:
    def test_calls_consolidate_psur_tool_with_correct_args(self, fake_graph):
        full_result = _consolidation_result(n_signals=2)
        responses = [
            _tool_call_message("consolidate_psur_tool",
                                {"drug_name": "KEYTRUDA", "period": "6mo"}, call_id="call_1"),
            AIMessage(content="Here is the KEYTRUDA 6-month PSUR summary."),
        ]
        fake_graph(responses)

        with patch("adverse_score.agent_tools.consolidate_psur") as mock_consolidate:
            mock_consolidate.return_value = full_result
            result = run_agent_turn([], "Consolidate KEYTRUDA for 6 months.")

        mock_consolidate.assert_called_once_with("KEYTRUDA", "6mo")
        assert isinstance(result, AgentTurnResult)
        assert result.dataset_changed is True
        assert result.updated_dataset is full_result
        assert "KEYTRUDA" in result.response_text

    def test_calls_resolve_drug_identity_tool_with_correct_args(self, fake_graph):
        identity = _drug_identity()
        responses = [
            _tool_call_message("resolve_drug_identity_tool",
                                {"drug_name": "KEYTRUDA"}, call_id="call_1"),
            AIMessage(content="KEYTRUDA resolves to PEMBROLIZUMAB."),
        ]
        fake_graph(responses)

        with patch("adverse_score.agent_tools.resolve_drug_identity") as mock_resolve:
            mock_resolve.return_value = identity
            result = run_agent_turn([], "What is KEYTRUDA's generic name?")

        mock_resolve.assert_called_once_with("KEYTRUDA")
        # Identity lookup alone never populates a dataset.
        assert result.dataset_changed is False
        assert result.updated_dataset is None

    def test_consolidation_error_surfaces_as_graceful_response_not_crash(self, fake_graph):
        responses = [
            _tool_call_message("consolidate_psur_tool",
                                {"drug_name": "BADDRUG", "period": "6mo"}, call_id="call_1"),
            AIMessage(content="I couldn't resolve that drug's identity."),
        ]
        fake_graph(responses)

        with patch("adverse_score.agent_tools.consolidate_psur") as mock_consolidate:
            mock_consolidate.return_value = ConsolidationError(
                drug_name="BADDRUG", stage="IDENTITY_RESOLUTION", reason="NOT_FOUND",
                message="No FDA-registered drug found matching 'BADDRUG'.",
            )
            result = run_agent_turn([], "Consolidate BADDRUG for 6 months.")

        assert isinstance(result, AgentTurnResult)
        assert result.dataset_changed is False
        assert result.updated_dataset is None

    def test_drug_identity_error_surfaces_as_graceful_response_not_crash(self, fake_graph):
        responses = [
            _tool_call_message("resolve_drug_identity_tool",
                                {"drug_name": "NOTADRUG"}, call_id="call_1"),
            AIMessage(content="I couldn't find that drug."),
        ]
        fake_graph(responses)

        with patch("adverse_score.agent_tools.resolve_drug_identity") as mock_resolve:
            mock_resolve.return_value = DrugIdentityError(
                query="NOTADRUG", reason="NOT_FOUND", message="No drug found.",
                attempted_variants=[],
            )
            result = run_agent_turn([], "What is NOTADRUG?")

        assert isinstance(result, AgentTurnResult)
        assert "couldn't find" in result.response_text.lower() or result.response_text


# ── AgentTurnResult dataset carry-forward correctness ──────────────────────

class TestDatasetCarryForward:
    def test_no_tool_call_carries_forward_in_session_dataset_unchanged(self, fake_graph):
        existing_dataset = _consolidation_result(n_signals=4)
        fake_graph([AIMessage(content="Sure, based on the current dataset, nausea ranks highest.")])

        result = run_agent_turn([], "What's the top signal?", in_session_dataset=existing_dataset)

        assert result.dataset_changed is False
        assert result.updated_dataset is existing_dataset

    def test_no_tool_call_and_no_prior_dataset_stays_none(self, fake_graph):
        fake_graph([AIMessage(content="I can help you consolidate PSUR data for a drug.")])

        result = run_agent_turn([], "What can you do?", in_session_dataset=None)

        assert result.dataset_changed is False
        assert result.updated_dataset is None

    def test_successful_consolidation_replaces_prior_dataset(self, fake_graph):
        old_dataset = _consolidation_result(n_signals=1, period="1yr")
        new_dataset = _consolidation_result(n_signals=9, period="6mo")
        responses = [
            _tool_call_message("consolidate_psur_tool",
                                {"drug_name": "HUMIRA", "period": "6mo"}, call_id="call_1"),
            AIMessage(content="Done consolidating HUMIRA."),
        ]
        fake_graph(responses)

        with patch("adverse_score.agent_tools.consolidate_psur") as mock_consolidate:
            mock_consolidate.return_value = new_dataset
            result = run_agent_turn([], "Now consolidate HUMIRA 6mo.", in_session_dataset=old_dataset)

        assert result.dataset_changed is True
        assert result.updated_dataset is new_dataset
        assert result.updated_dataset is not old_dataset


# ── AGENT_MAX_ITERATIONS enforcement ────────────────────────────────────────

class TestIterationCap:
    def test_agent_terminates_within_configured_cap(self, fake_graph, monkeypatch):
        """A fake LLM that ALWAYS wants to call a tool must not loop forever —
        run_agent_turn must return gracefully once AGENT_MAX_ITERATIONS is exceeded."""
        # Unique call ids per response — required so langgraph doesn't collapse
        # cycling duplicate-id tool calls into a corrupted graph state.
        responses = [
            _tool_call_message("resolve_drug_identity_tool", {"drug_name": "X"}, call_id=f"call_{i}")
            for i in range(50)
        ]
        fake_graph(responses)

        with patch("adverse_score.agent_tools.resolve_drug_identity") as mock_resolve:
            mock_resolve.return_value = DrugIdentityError(
                query="X", reason="NOT_FOUND", message="not found", attempted_variants=[])
            result = run_agent_turn([], "Keep looking up X forever.")

        assert isinstance(result, AgentTurnResult)
        assert isinstance(result.response_text, str)
        assert len(result.response_text) > 0
        # Bounded call count proves it didn't spin unboundedly.
        assert mock_resolve.call_count <= orchestrator.AGENT_MAX_ITERATIONS + 1


# ── Layering: no persistence dependency ────────────────────────────────────

class TestNoPersistenceImport:
    def test_source_has_no_persistence_or_store_import(self):
        """AST-based check (not a naive substring scan) — the module docstring
        legitimately mentions "ConsolidationStore" in prose explaining the
        design constraint; what matters is that no actual import statement
        pulls in persistence.py or ConsolidationStore."""
        import ast
        import inspect

        source = inspect.getsource(orchestrator)
        tree = ast.parse(source)

        imported_modules = []
        imported_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.append(node.module)
                imported_names.extend(alias.name for alias in node.names)

        assert not any("persistence" in m for m in imported_modules)
        assert "ConsolidationStore" not in imported_names


# ── Import safety ────────────────────────────────────────────────────────

class TestImportSafety:
    def test_import_succeeds_without_api_keys(self, monkeypatch):
        """`from adverse_score.orchestrator import run_agent_turn, AgentTurnResult`
        succeeds without raising even when OPENAI_API_KEY/OPENFDA_API_KEY are absent —
        only CALLING run_agent_turn() should require keys."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENFDA_API_KEY", raising=False)

        import importlib
        from adverse_score import orchestrator as orch_module
        importlib.reload(orch_module)

        assert callable(orch_module.run_agent_turn)
        assert orch_module.AgentTurnResult is not None
