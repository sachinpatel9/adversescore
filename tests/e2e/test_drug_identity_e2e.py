"""
End-to-end integration tests for drug_identity.py — live openFDA API
resolution (label.json / ndc.json / drugsfda.json).

Run:
    pytest tests/e2e/test_drug_identity_e2e.py -v -m e2e

Skip when API keys are absent:
    Tests auto-skip via SKIP_NO_FDA markers in conftest.py.
"""

import pytest

# All tests in this file are E2E
pytestmark = pytest.mark.e2e

# Import skip guards from conftest
from conftest import SKIP_NO_FDA

from adverse_score.drug_identity import DrugIdentity, DrugIdentityError, resolve_drug_identity


@SKIP_NO_FDA
class TestDrugIdentityE2E:
    """Live resolution against real openFDA data."""

    def test_resolve_known_drug(self, e2e_client):
        """A stable, well-known prescription drug resolves with non-empty name
        variants and an approval date. Deliberately not an OTC monograph drug
        (e.g. aspirin/ibuprofen) — those carry an OTC monograph number rather
        than an NDA/ANDA/BLA and have no drugsfda.json entry, so their
        market_authorization_date is legitimately None (a documented
        limitation of the drugsfda.json-based approach, not a bug)."""
        result = resolve_drug_identity("KEYTRUDA", client=e2e_client.fda)
        assert isinstance(result, DrugIdentity)
        assert result.brand_names or result.generic_names
        assert result.market_authorization_date is not None

    def test_resolve_unknown_drug(self, e2e_client):
        """A nonsense drug name produces a clean structured NOT_FOUND error."""
        result = resolve_drug_identity("ZZZNOTADRUGZZZ123", client=e2e_client.fda)
        assert isinstance(result, DrugIdentityError)
        assert result.reason == "NOT_FOUND"
