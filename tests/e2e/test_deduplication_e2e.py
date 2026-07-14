"""
End-to-end integration test for deduplication.py (Phase 3) — runs live-retrieved
FAERS data through deduplicate_reports() to validate structural invariants against
real data shapes, not just mocks.
"""
import pytest
pytestmark = pytest.mark.e2e
from conftest import SKIP_NO_FDA
from adverse_score.deduplication import deduplicate_reports
from datetime import date


@SKIP_NO_FDA
class TestDeduplicationE2E:
    def test_live_psur_retrieval_dedup(self, e2e_client):
        """KEYTRUDA, 6mo period — smallest period, minimal live HTTP calls."""
        result = e2e_client.fda.fetch_psur_reports(["KEYTRUDA"], date(2014, 9, 4), "6mo")
        dedup_result = deduplicate_reports(result.reports)
        assert dedup_result.total_output_count <= dedup_result.total_input_count
        assert len(dedup_result.reports) == dedup_result.total_output_count
        assert dedup_result.total_input_count == len(result.reports)
        assert len(dedup_result.audit_trail) == (dedup_result.removed_by_exact_id + dedup_result.removed_by_heuristic)
