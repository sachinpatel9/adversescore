"""
End-to-end integration tests for time-bounded PRR — validates that
_calculate_prr_metrics accepts date bounds against live openFDA data.

Run:
    pytest tests/e2e/test_prr_e2e.py -v -m e2e

Skip when API keys are absent:
    Tests auto-skip via SKIP_NO_FDA markers in conftest.py.
"""

import pytest

# All tests in this file are E2E
pytestmark = pytest.mark.e2e

# Import skip guards from conftest
from conftest import SKIP_NO_FDA


@SKIP_NO_FDA
class TestTimeBoundedPRR:
    """Validates that _calculate_prr_metrics accepts date bounds (Fix 2C)."""

    def test_prr_metrics_with_date_bounds(self, e2e_client):
        """_calculate_prr_metrics with start_date/end_date returns valid structure or None."""
        metrics = e2e_client._calculate_prr_metrics(
            drug_name="aspirin",
            pharm_class="",
            target_symptom="nausea",
            start_date="20240101",
            end_date="20241231",
        )
        assert metrics is None or isinstance(metrics, dict), (
            f"Expected None or dict, got {type(metrics)}"
        )
        if metrics is not None:
            assert "prr" in metrics
            assert "signal_detected" in metrics
            assert isinstance(metrics["signal_detected"], bool)

    def test_prr_date_bounds_narrower_than_all_time(self, e2e_client):
        """A one-year window should return fewer or equal drug cases than all-time."""
        all_time = e2e_client._calculate_prr_metrics(
            drug_name="metformin",
            pharm_class="",
            target_symptom="nausea",
        )
        bounded = e2e_client._calculate_prr_metrics(
            drug_name="metformin",
            pharm_class="",
            target_symptom="nausea",
            start_date="20240101",
            end_date="20241231",
        )
        if all_time is None or bounded is None:
            pytest.skip("Insufficient data for PRR comparison")
        assert bounded["drug_cases"] <= all_time["drug_cases"], (
            "Bounded window returned more cases than all-time — date filter not applied"
        )
