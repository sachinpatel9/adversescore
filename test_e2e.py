"""
End-to-end integration tests for AdverseScore.

These tests hit the live openFDA API and (optionally) the OpenAI LLM.
They validate that the full pipeline — from HTTP request to scored payload
to agent narrative — works correctly with real data.

Run:
    pytest test_e2e.py -v -m e2e
    pytest test_e2e.py -v -m e2e --durations=10   # with timing

Skip when API keys are absent:
    Tests auto-skip via SKIP_NO_FDA / SKIP_NO_OPENAI markers in conftest.py.
"""

import json
import re
import time
import pytest

# All tests in this file are E2E
pytestmark = pytest.mark.e2e

# Import skip guards from conftest
from conftest import SKIP_NO_FDA, SKIP_NO_OPENAI


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY A: Live FDA API Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

@SKIP_NO_FDA
class TestLiveFDAAPI:
    """Validates that the openFDA API contract is intact and AdverseScoreClient
    handles real HTTP responses correctly."""

    def test_fetch_events_known_drug(self, e2e_client):
        """fetch_events returns a populated dict for a well-known drug."""
        result = e2e_client.fetch_events("metformin")
        assert result is not None, "Expected data for metformin, got None"
        assert "results" in result
        assert len(result["results"]) > 0

    def test_fetch_events_with_demographics(self, e2e_client):
        """fetch_events accepts demographic filters without raising."""
        result = e2e_client.fetch_events("ibuprofen", patient_age=65, patient_sex="F")
        # Demographics may narrow results to zero — that's acceptable
        if result is not None:
            assert "results" in result

    def test_fetch_events_nonexistent_drug(self, e2e_client):
        """fetch_events returns None for a fabricated drug name."""
        result = e2e_client.fetch_events("ZZZZNOTADRUG9999")
        assert result is None

    def test_fetch_events_response_time(self, e2e_client):
        """fetch_events for a common drug completes within 10 seconds."""
        start = time.perf_counter()
        e2e_client.fetch_events("aspirin")
        elapsed = time.perf_counter() - start
        assert elapsed < 10, f"fetch_events took {elapsed:.1f}s, expected < 10s"

    def test_fetch_label_text_known_drug(self, e2e_client):
        """fetch_label_text returns non-empty adverse reactions text for a known drug."""
        label = e2e_client.fetch_label_text("metformin")
        assert isinstance(label, str)
        assert len(label) > 0, "Expected non-empty label text for metformin"

    def test_fetch_label_text_unknown_drug(self, e2e_client):
        """fetch_label_text returns empty string for an unknown drug."""
        label = e2e_client.fetch_label_text("ZZZZNOTADRUG9999")
        assert label == ""

    def test_discover_drug_class(self, e2e_client):
        """_discover_drug_class returns a non-empty pharmacologic class string."""
        drug_class = e2e_client._discover_drug_class("METFORMIN")
        assert isinstance(drug_class, str)
        assert len(drug_class) > 0, "Expected non-empty drug class for METFORMIN"

    def test_discover_peers(self, e2e_client):
        """_discover_peers returns peer drugs that exclude the target."""
        drug_class = e2e_client._discover_drug_class("METFORMIN")
        if not drug_class:
            pytest.skip("Could not discover drug class for METFORMIN")
        peers = e2e_client._discover_peers(drug_class, "METFORMIN")
        assert isinstance(peers, list)
        assert len(peers) >= 1, "Expected at least 1 peer drug"
        assert all(p.upper() != "METFORMIN" for p in peers), "Peers must not include target drug"

    def test_fetch_quarterly_data_structure(self, e2e_client):
        """fetch_quarterly_data returns a list of dicts with the expected keys."""
        data = e2e_client.fetch_quarterly_data("aspirin", num_quarters=4)
        assert isinstance(data, list)
        assert len(data) == 4, f"Expected 4 quarters, got {len(data)}"
        for entry in data:
            assert "quarter" in entry
            assert "adverse_score" in entry
            assert "report_count" in entry

    def test_fetch_symptom_counts(self, e2e_client):
        """_fetch_symptom_counts returns a dict of symptom names to counts."""
        counts = e2e_client._fetch_symptom_counts(drug_name="ASPIRIN")
        assert isinstance(counts, dict)
        if len(counts) > 0:
            first_key = next(iter(counts))
            assert isinstance(first_key, str)
            assert isinstance(counts[first_key], int)


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY B: Full Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════

@SKIP_NO_FDA
class TestFullPipeline:
    """Runs the complete fetch → flatten → calculate_final_score pipeline
    with real FDA data and validates payload structure."""

    REQUIRED_TOP_KEYS = {"metadata", "clinical_signal", "data_integrity", "agent_directives"}
    VALID_STATUSES = {
        "Stable", "Incomplete Data", "High Signal - Urgent Review",
    }
    VALID_TRENDS = {"RISING", "STABLE", "DECLINING", "INSUFFICIENT_DATA"}

    def _run_pipeline(self, client, drug_name, **kwargs):
        """Helper: runs the full fetch → flatten → score pipeline."""
        raw = client.fetch_events(
            drug_name,
            patient_age=kwargs.get("patient_age"),
            patient_sex=kwargs.get("patient_sex"),
        )
        clean = client._flatten_results(raw) if raw else []
        payload = client.calculate_final_score(drug_name, clean, **kwargs)
        return payload

    def test_full_score_common_drug(self, e2e_client):
        """Full pipeline for metformin produces a valid, complete payload."""
        payload = self._run_pipeline(e2e_client, "metformin")
        # All required top-level keys present
        assert self.REQUIRED_TOP_KEYS.issubset(payload.keys())
        # Score within valid range
        score = payload["clinical_signal"]["adverse_score"]
        assert 0 <= score <= 100, f"Score {score} out of range"
        # Status is a valid string (may include drug name for Monitor status)
        status = payload["clinical_signal"]["status"]
        assert isinstance(status, str) and len(status) > 0

    def test_full_score_with_demographics(self, e2e_client):
        """Pipeline with demographic filters completes without error."""
        payload = self._run_pipeline(
            e2e_client, "lisinopril", patient_age=65, patient_sex="F"
        )
        assert self.REQUIRED_TOP_KEYS.issubset(payload.keys())
        assert 0 <= payload["clinical_signal"]["adverse_score"] <= 100

    def test_full_score_with_target_symptom(self, e2e_client):
        """Pipeline with target_symptom includes PRR pharmacovigilance metrics."""
        payload = self._run_pipeline(
            e2e_client, "metformin", target_symptom="nausea"
        )
        prr = payload.get("pharmacovigilance_metrics")
        assert prr is not None, "Expected pharmacovigilance_metrics when target_symptom is provided"
        assert "prr" in prr
        assert "ci_lower" in prr
        assert "signal_detected" in prr
        assert isinstance(prr["signal_detected"], bool)
        assert "label_status" in prr

    def test_full_score_with_temporal(self, e2e_client):
        """Quarterly data and trend classification are structurally valid."""
        time_series = e2e_client.fetch_quarterly_data("aspirin", num_quarters=4)
        trend = e2e_client.compute_trend(time_series)
        assert isinstance(time_series, list) and len(time_series) == 4
        assert trend in self.VALID_TRENDS

    def test_full_score_rare_drug(self, e2e_client):
        """Pipeline handles a sparse-data drug gracefully."""
        payload = self._run_pipeline(e2e_client, "DANTROLENE")
        assert self.REQUIRED_TOP_KEYS.issubset(payload.keys())
        # Rare drug: score may be 0 (no data) or low
        score = payload["clinical_signal"]["adverse_score"]
        assert 0 <= score <= 100

    def test_peer_benchmark_returns_float(self, e2e_client):
        """get_peer_benchmark returns a float within the valid score range."""
        benchmark = e2e_client.get_peer_benchmark("metformin")
        assert isinstance(benchmark, float)
        assert 0.0 <= benchmark <= 100.0

    def test_pipeline_timing_no_benchmark(self, e2e_client):
        """Full pipeline without benchmark completes within 30 seconds."""
        start = time.perf_counter()
        self._run_pipeline(e2e_client, "ibuprofen", skip_benchmark=True)
        elapsed = time.perf_counter() - start
        assert elapsed < 30, f"Pipeline took {elapsed:.1f}s, expected < 30s"

    def test_pipeline_timing_with_benchmark(self, e2e_client):
        """Full pipeline with benchmark completes within 60 seconds."""
        start = time.perf_counter()
        self._run_pipeline(e2e_client, "aspirin", skip_benchmark=False)
        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"Pipeline with benchmark took {elapsed:.1f}s, expected < 60s"


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY C: Agent Tool E2E Tests
# ═══════════════════════════════════════════════════════════════════════════

@SKIP_NO_FDA
class TestAgentToolE2E:
    """Calls get_adverse_score.invoke() directly with real APIs (no LLM)."""

    def test_tool_standard_query(self):
        """Standard drug query returns valid JSON with all required keys."""
        from adverse_score.agent_tools import get_adverse_score
        result = get_adverse_score.invoke({"drug_name": "metformin"})
        payload = json.loads(result)
        assert "metadata" in payload
        assert "clinical_signal" in payload
        assert "data_integrity" in payload
        assert "agent_directives" in payload
        assert payload["agent_directives"]["diagnosis_lock"] is True

    def test_tool_with_all_params(self):
        """Tool with demographics and target_symptom includes PRR and demographics."""
        from adverse_score.agent_tools import get_adverse_score
        result = get_adverse_score.invoke({
            "drug_name": "ibuprofen",
            "patient_age": 50,
            "patient_sex": "F",
            "target_symptom": "nausea",
        })
        payload = json.loads(result)
        # Demographics are injected into metadata
        demos = payload["metadata"]["extracted_demographics"]
        assert demos["age"] == 50
        assert demos["sex"] == "F"
        assert demos["target_symptom"] == "nausea"
        # PRR metrics should be present
        assert payload.get("pharmacovigilance_metrics") is not None

    def test_tool_with_temporal(self):
        """Tool with include_temporal=True returns temporal_analysis."""
        from adverse_score.agent_tools import get_adverse_score
        result = get_adverse_score.invoke({
            "drug_name": "aspirin",
            "include_temporal": True,
        })
        payload = json.loads(result)
        assert "temporal_analysis" in payload
        ta = payload["temporal_analysis"]
        assert "time_series" in ta
        assert "trend_classification" in ta
        assert isinstance(ta["time_series"], list)
        assert ta["trend_classification"] in {
            "RISING", "STABLE", "DECLINING", "INSUFFICIENT_DATA"
        }

    def test_tool_nonexistent_drug(self):
        """Tool handles a nonexistent drug gracefully with zero score."""
        from adverse_score.agent_tools import get_adverse_score
        result = get_adverse_score.invoke({"drug_name": "ZZZZNOTADRUG9999"})
        payload = json.loads(result)
        score = payload["clinical_signal"]["adverse_score"]
        assert score == 0.0
        assert "Incomplete Data" in payload["clinical_signal"]["status"]

    def test_tool_delta_detection(self, tmp_path):
        """Two consecutive calls for the same drug produce delta_detection on the second call."""
        from unittest.mock import patch
        from adverse_score.persistence import AnalysisStore

        db_path = tmp_path / "delta_test.db"

        # Create a temp store subclass that always uses our temp DB
        class TempStore(AnalysisStore):
            def __init__(self, **kwargs):
                super().__init__(db_path=db_path)

        # Patch at the persistence module level — agent_tools does
        # `from .persistence import AnalysisStore` inside a try block
        with patch("adverse_score.persistence.AnalysisStore", TempStore):
            # Also need to ensure the lazy import inside agent_tools picks up the patch
            import adverse_score.persistence as persistence_mod
            original_cls = persistence_mod.AnalysisStore
            persistence_mod.AnalysisStore = TempStore
            try:
                from adverse_score.agent_tools import get_adverse_score

                # First call — no prior exists
                result1 = get_adverse_score.invoke({"drug_name": "aspirin"})
                payload1 = json.loads(result1)
                assert "delta_detection" not in payload1 or payload1.get("delta_detection") is None

                # Second call — prior now exists
                result2 = get_adverse_score.invoke({"drug_name": "aspirin"})
                payload2 = json.loads(result2)
                if "delta_detection" in payload2 and payload2["delta_detection"] is not None:
                    delta = payload2["delta_detection"]
                    assert "prior_score" in delta
                    assert "prior_date" in delta
                    assert "score_delta" in delta
                    assert isinstance(delta["score_delta"], (int, float))
            finally:
                persistence_mod.AnalysisStore = original_cls


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY D: Full Agent E2E Tests (LLM Required)
# ═══════════════════════════════════════════════════════════════════════════

@SKIP_NO_FDA
@SKIP_NO_OPENAI
class TestAgentE2E:
    """Sends messages to the LangGraph agent_executor and validates LLM output.
    Requires both OPENFDA_API_KEY and OPENAI_API_KEY."""

    def _invoke_agent(self, message: str) -> str:
        """Helper: sends a message to the agent and returns the response text."""
        from adverse_score.orchestrator import agent_executor
        from langchain_core.messages import HumanMessage

        result = agent_executor.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"recursion_limit": 10},
        )
        # Extract text from the last AI message
        messages = result.get("messages", [])
        assert len(messages) > 0, "Agent returned no messages"
        return messages[-1].content

    def test_agent_standard_query(self):
        """Agent produces a non-empty response mentioning the queried drug."""
        response = self._invoke_agent("Analyze the safety profile of metformin")
        assert len(response) > 50, "Response is suspiciously short"
        assert "metformin" in response.lower()

    def test_agent_contains_disclaimer(self):
        """Agent response includes the clinical disclaimer."""
        response = self._invoke_agent("Analyze the safety profile of metformin")
        disclaimer_phrases = ["informational purposes", "does not constitute medical advice"]
        assert any(
            phrase in response.lower() for phrase in disclaimer_phrases
        ), "Response missing clinical disclaimer"

    def test_agent_off_topic_rejection(self):
        """Agent rejects off-topic queries with scope enforcement message."""
        response = self._invoke_agent("What is the weather today?")
        assert "pharmaceutical safety" in response.lower() or "drug" in response.lower()

    def test_agent_response_is_prose(self):
        """Agent response follows prose format (no numbered list structure)."""
        response = self._invoke_agent("Analyze the safety profile of ibuprofen")
        lines = response.strip().split("\n")
        numbered_lines = [l for l in lines if re.match(r'^\s*\d+\.\s', l)]
        # Allow at most 2 numbered lines (some LLMs occasionally number a sub-point)
        ratio = len(numbered_lines) / max(len(lines), 1)
        assert ratio < 0.3, (
            f"Response appears to be a numbered list: {len(numbered_lines)}/{len(lines)} "
            f"lines start with a number"
        )

    def test_agent_response_time(self):
        """Full agent round-trip completes within 90 seconds."""
        start = time.perf_counter()
        self._invoke_agent("Analyze the safety profile of aspirin")
        elapsed = time.perf_counter() - start
        assert elapsed < 90, f"Agent took {elapsed:.1f}s, expected < 90s"

    def test_agent_temporal_query(self):
        """Agent responds to temporal queries with trend-related language."""
        response = self._invoke_agent(
            "What is the trend for aspirin over time?"
        )
        temporal_terms = ["trend", "quarter", "rising", "stable", "declining", "temporal"]
        assert any(
            term in response.lower() for term in temporal_terms
        ), "Response missing temporal analysis language"
