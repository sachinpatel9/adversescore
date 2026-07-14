"""
Unit tests for agent_tools.py (Phase 8) — LangChain tool wrappers around
drug_identity.resolve_drug_identity and consolidation.consolidate_psur.

All FDA/OpenAI-hitting internals are mocked; these tests never make network
calls. Tool functions are `@tool`-decorated, so they're invoked via
`.invoke({...})` (the LangChain BaseTool calling convention) rather than as
plain Python callables.
"""
from datetime import date
from unittest.mock import patch

import pytest

from adverse_score import agent_tools
from adverse_score.consolidation import ConsolidationError, ConsolidationResult
from adverse_score.deduplication import DedupResult
from adverse_score.drug_identity import DrugIdentity, DrugIdentityError
from adverse_score.fda_client import ChunkResult, PSURRetrievalResult
from adverse_score.label_classifier import LabelClassificationResult
from adverse_score.ranking import RankedSignal, RankingResult


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


def _ranked_signal(symptom="NAUSEA", rank=1, report_count=5, **overrides):
    defaults = dict(
        symptom=symptom,
        rank=rank,
        seriousness_tier="NON_SERIOUS",
        strength_of_evidence_tier="WEAK_UNLABELED",
        reversibility_tier="UNKNOWN",
        public_health_tier="LOW",
        prr_metrics={
            "prr": 1.2, "ci_lower": 0.9, "signal_detected": False,
            "target_symptom": symptom, "drug_cases": report_count,
            "class_cases": 10, "label_status": "UNLABELED",
        },
        report_count=report_count,
    )
    defaults.update(overrides)
    return RankedSignal(**defaults)


def _consolidation_result(n_signals=25, **overrides):
    identity = _drug_identity()
    chunk = ChunkResult(
        label="chunk-0-20240101-to-20240401", start_date="20240101", end_date="20240401",
        reports=[], retrieved_count=100, estimated_total_count=100, truncated=False, error=None,
    )
    retrieval = PSURRetrievalResult(
        canonical_query_variants=["KEYTRUDA"], period_start="20240101", period_end="20240701",
        chunks=[chunk], reports=[], total_retrieved_count=100, total_estimated_count=100,
        any_chunk_truncated=False, used_fallback_anchor=False, fallback_reason=None,
    )
    dedup = DedupResult(
        reports=[], total_input_count=100, total_output_count=90,
        removed_by_exact_id=8, removed_by_heuristic=2, audit_trail=[],
    )
    ranked_signals = [_ranked_signal(symptom=f"SIGNAL_{i}", rank=i + 1) for i in range(n_signals)]
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


@pytest.fixture(autouse=True)
def _fresh_capture_box():
    """Installs a fresh capture box before every test (mirroring what
    orchestrator.run_agent_turn does before invoking the graph — see
    agent_tools.py's module docstring on why begin_consolidation_capture()
    must run BEFORE a tool's .invoke(), not lazily inside the tool) and
    clears it afterward so no captured result leaks between tests."""
    agent_tools.begin_consolidation_capture()
    yield
    agent_tools.pop_last_consolidation_result()


# ── resolve_drug_identity_tool ──────────────────────────────────────────────

class TestResolveDrugIdentityTool:
    def test_success_returns_json_safe_dict_with_isoformat_date(self):
        with patch.object(agent_tools, "resolve_drug_identity") as mock_resolve:
            mock_resolve.return_value = _drug_identity()
            result = agent_tools.resolve_drug_identity_tool.invoke({"drug_name": "KEYTRUDA"})

        mock_resolve.assert_called_once_with("KEYTRUDA")
        assert result["canonical_name"] == "KEYTRUDA"
        assert result["market_authorization_date"] == "2014-09-04"
        assert isinstance(result["market_authorization_date"], str)
        assert "error" not in result

    def test_success_with_none_market_authorization_date(self):
        """OTC monograph drugs (PARTIAL confidence) have market_authorization_date=None —
        must serialize to None, not crash on .isoformat()."""
        with patch.object(agent_tools, "resolve_drug_identity") as mock_resolve:
            mock_resolve.return_value = _drug_identity(
                market_authorization_date=None,
                market_authorization_date_source="no AP-status submission found",
                resolution_confidence="PARTIAL",
            )
            result = agent_tools.resolve_drug_identity_tool.invoke({"drug_name": "ASPIRIN"})

        assert result["market_authorization_date"] is None
        assert result["resolution_confidence"] == "PARTIAL"

    def test_error_returns_graceful_dict_not_crash(self):
        with patch.object(agent_tools, "resolve_drug_identity") as mock_resolve:
            mock_resolve.return_value = DrugIdentityError(
                query="NOTADRUG", reason="NOT_FOUND",
                message="No FDA-registered drug found matching 'NOTADRUG'.",
                attempted_variants=["label.json exact brand/generic: NOTADRUG"],
            )
            result = agent_tools.resolve_drug_identity_tool.invoke({"drug_name": "NOTADRUG"})

        assert result == {
            "error": "NOT_FOUND",
            "message": "No FDA-registered drug found matching 'NOTADRUG'.",
        }

    def test_unexpected_exception_returns_graceful_dict_not_crash(self):
        with patch.object(agent_tools, "resolve_drug_identity") as mock_resolve:
            mock_resolve.side_effect = RuntimeError("boom")
            result = agent_tools.resolve_drug_identity_tool.invoke({"drug_name": "KEYTRUDA"})

        assert result["error"] == "UNEXPECTED_EXCEPTION"


# ── consolidate_psur_tool ───────────────────────────────────────────────────

class TestConsolidatePsurTool:
    def test_success_returns_bounded_summary_and_stashes_full_result(self):
        full_result = _consolidation_result(n_signals=25)
        with patch.object(agent_tools, "consolidate_psur") as mock_consolidate:
            mock_consolidate.return_value = full_result
            summary = agent_tools.consolidate_psur_tool.invoke(
                {"drug_name": "KEYTRUDA", "period": "6mo"})

        mock_consolidate.assert_called_once_with("KEYTRUDA", "6mo")

        # Bounded to TOP_N_NARRATED_SIGNALS even though 25 signals exist.
        from adverse_score.config import TOP_N_NARRATED_SIGNALS
        assert len(summary["top_signals"]) == TOP_N_NARRATED_SIGNALS
        assert summary["total_signals"] == 25
        assert summary["canonical_name"] == "KEYTRUDA"
        assert summary["period"] == "6mo"
        assert summary["total_retrieved_count"] == 100
        assert summary["total_estimated_count"] == 100
        assert summary["any_chunk_truncated"] is False
        assert summary["dedup_total_input_count"] == 100
        assert summary["dedup_total_output_count"] == 90
        assert summary["dedup_removed_by_exact_id"] == 8
        assert summary["dedup_removed_by_heuristic"] == 2
        assert summary["formula_version"] == "1.0"

        # No raw report lists / full nested dataclasses leaked into the summary.
        assert "reports" not in summary
        assert "ranked_signals" not in summary

        # Full result captured out-of-band, independent of the summary dict.
        captured = agent_tools.pop_last_consolidation_result()
        assert captured is full_result

    def test_summary_is_json_serializable(self):
        import json
        full_result = _consolidation_result(n_signals=3)
        with patch.object(agent_tools, "consolidate_psur") as mock_consolidate:
            mock_consolidate.return_value = full_result
            summary = agent_tools.consolidate_psur_tool.invoke(
                {"drug_name": "KEYTRUDA", "period": "6mo"})
        json.dumps(summary)  # must not raise

    def test_error_returns_graceful_dict_not_crash_and_does_not_stash(self):
        with patch.object(agent_tools, "consolidate_psur") as mock_consolidate:
            mock_consolidate.return_value = ConsolidationError(
                drug_name="KEYTRUDA", stage="RETRIEVAL", reason="NO_NAME_VARIANTS",
                message="No queryable name variants resolved.",
            )
            result = agent_tools.consolidate_psur_tool.invoke(
                {"drug_name": "KEYTRUDA", "period": "6mo"})

        assert result == {
            "error": "RETRIEVAL",
            "reason": "NO_NAME_VARIANTS",
            "message": "No queryable name variants resolved.",
        }
        assert agent_tools.pop_last_consolidation_result() is None

    def test_invalid_period_returns_graceful_dict_without_calling_pipeline(self):
        """Regression test: a live LLM run during Phase 8 build passed
        period="last 6 months" (natural language) instead of the required
        literal "6mo" — this used to raise an uncaught KeyError deep inside
        fda_client.py's compute_psur_period(), crashing the whole graph run.
        The tool must validate period BEFORE calling consolidate_psur()."""
        with patch.object(agent_tools, "consolidate_psur") as mock_consolidate:
            result = agent_tools.consolidate_psur_tool.invoke(
                {"drug_name": "KEYTRUDA", "period": "last 6 months"})

        mock_consolidate.assert_not_called()
        assert result["error"] == "INVALID_PERIOD"
        assert "6mo" in result["message"]
        assert agent_tools.pop_last_consolidation_result() is None

    def test_unexpected_exception_from_pipeline_returns_graceful_dict(self):
        """Any exception escaping consolidate_psur() (a real pipeline bug,
        not a structured ConsolidationError) must not crash the tool call —
        LangGraph's create_agent (this langchain version) offers no way to
        catch tool exceptions itself, so the tool is the last line of
        defense."""
        with patch.object(agent_tools, "consolidate_psur") as mock_consolidate:
            mock_consolidate.side_effect = KeyError("boom")
            result = agent_tools.consolidate_psur_tool.invoke(
                {"drug_name": "KEYTRUDA", "period": "6mo"})

        assert result["error"] == "UNEXPECTED_EXCEPTION"
        assert agent_tools.pop_last_consolidation_result() is None

    def test_pop_clears_cache_after_read(self):
        full_result = _consolidation_result(n_signals=1)
        with patch.object(agent_tools, "consolidate_psur") as mock_consolidate:
            mock_consolidate.return_value = full_result
            agent_tools.consolidate_psur_tool.invoke({"drug_name": "KEYTRUDA", "period": "6mo"})

        assert agent_tools.pop_last_consolidation_result() is full_result
        assert agent_tools.pop_last_consolidation_result() is None  # second pop is empty

    def test_no_pre_installed_capture_box_is_a_safe_no_op(self):
        """If begin_consolidation_capture() was never called, the tool must
        still run successfully (returning its normal summary dict) — only
        the out-of-band full-result capture is silently skipped, per the
        documented no-op behavior in agent_tools.py's module docstring."""
        agent_tools._capture_var.set(None)  # simulate "capture never begun"
        full_result = _consolidation_result(n_signals=1)
        with patch.object(agent_tools, "consolidate_psur") as mock_consolidate:
            mock_consolidate.return_value = full_result
            summary = agent_tools.consolidate_psur_tool.invoke(
                {"drug_name": "KEYTRUDA", "period": "6mo"})

        assert summary["canonical_name"] == "KEYTRUDA"
        assert agent_tools.pop_last_consolidation_result() is None
        assert agent_tools.pop_last_consolidation_result() is None  # second pop is empty


# ── Regression guards ────────────────────────────────────────────────────
# The OLD `get_adverse_score`/`_global_client` symbols (Phase 0 removal) must
# still not exist. The NEW Phase 8 tool symbols must exist with the right
# names/types.

class TestAgentToolsModuleShape:
    def test_get_adverse_score_removed(self):
        assert not hasattr(agent_tools, "get_adverse_score")

    def test_global_client_removed(self):
        assert not hasattr(agent_tools, "_global_client")

    def test_new_tools_exist_and_are_langchain_tools(self):
        from langchain_core.tools import BaseTool
        assert isinstance(agent_tools.resolve_drug_identity_tool, BaseTool)
        assert isinstance(agent_tools.consolidate_psur_tool, BaseTool)
        assert agent_tools.resolve_drug_identity_tool.name == "resolve_drug_identity_tool"
        assert agent_tools.consolidate_psur_tool.name == "consolidate_psur_tool"

    def test_capture_helpers_exist(self):
        assert callable(agent_tools.pop_last_consolidation_result)
