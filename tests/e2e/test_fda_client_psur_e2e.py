"""
End-to-end integration tests for fda_client.py's PSUR chunked/paginated
retrieval (Phase 2) — live openFDA API validation.

Run:
    pytest tests/e2e/test_fda_client_psur_e2e.py -v -m e2e

Skip when API keys are absent:
    Tests auto-skip via SKIP_NO_FDA markers in conftest.py.
"""

from datetime import date

import pytest

# All tests in this file are E2E
pytestmark = pytest.mark.e2e

# Import skip guards from conftest
from conftest import SKIP_NO_FDA


@SKIP_NO_FDA
class TestPsurRetrievalE2E:
    """Live PSUR retrieval against real openFDA data. Asserts structural invariants
    rather than exact counts, since live data changes over time."""

    def test_known_drug_anchored_period(self, e2e_client):
        """KEYTRUDA (real approval date 2014-09-04, well over 6 months old) with a
        6mo period — smallest period, minimal live HTTP calls."""
        result = e2e_client.fda.fetch_psur_reports(
            ["KEYTRUDA"], date(2014, 9, 4), "6mo")

        report_ids = [r["report_id"] for r in result.reports]
        assert len(set(report_ids)) == len(report_ids), "Duplicate report_id in merged output"
        assert result.total_retrieved_count == len(result.reports)
        assert result.total_retrieved_count <= result.total_estimated_count
        assert len(result.chunks) == 2
        assert result.used_fallback_anchor is False

    def test_otc_drug_uses_fallback_anchor(self, e2e_client):
        """ASPIRIN is an OTC monograph drug (application number, e.g. "M013", not an
        NDA/ANDA/BLA) with no drugsfda.json entry — market_authorization_date is None
        per drug_identity.py's documented limitation, so this exercises the live
        fallback-anchoring path with genuine data shape, not just a mocked one."""
        result = e2e_client.fda.fetch_psur_reports(
            ["ASPIRIN"], None, "6mo")

        assert result.used_fallback_anchor is True
        assert result.fallback_reason == "NO_MARKET_AUTH_DATE"
        report_ids = [r["report_id"] for r in result.reports]
        assert len(set(report_ids)) == len(report_ids)
        assert result.total_retrieved_count == len(result.reports)
