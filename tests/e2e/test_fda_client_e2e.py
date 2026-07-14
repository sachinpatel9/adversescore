"""
End-to-end integration tests for fda_client.py — live openFDA API contract
validation.

Run:
    pytest tests/e2e/test_fda_client_e2e.py -v -m e2e

Skip when API keys are absent:
    Tests auto-skip via SKIP_NO_FDA markers in conftest.py.
"""

import time
import pytest

# All tests in this file are E2E
pytestmark = pytest.mark.e2e

# Import skip guards from conftest
from conftest import SKIP_NO_FDA


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

    def test_fetch_symptom_counts(self, e2e_client):
        """_fetch_symptom_counts returns a dict of symptom names to counts."""
        counts = e2e_client._fetch_symptom_counts(drug_name="ASPIRIN")
        assert isinstance(counts, dict)
        if len(counts) > 0:
            first_key = next(iter(counts))
            assert isinstance(first_key, str)
            assert isinstance(counts[first_key], int)
