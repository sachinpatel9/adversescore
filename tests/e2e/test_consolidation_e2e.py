"""
End-to-end integration test for consolidation.py (Phase 6) — runs the full
consolidate_psur() pipeline (drug identity resolution -> chunked FDA retrieval
-> deduplication -> label lookup -> ranking) against the live openFDA API.
"""
import pytest
pytestmark = pytest.mark.e2e
from conftest import SKIP_NO_FDA, SKIP_NO_OPENAI
from adverse_score.consolidation import consolidate_psur, ConsolidationResult
from adverse_score.config import (
    RANKING_FORMULA_VERSION,
    SERIOUSNESS_TIER_ORDER,
    STRENGTH_TIER_ORDER,
    REVERSIBILITY_TIER_ORDER,
    PUBLIC_HEALTH_TIER_ORDER,
)


@SKIP_NO_FDA
@SKIP_NO_OPENAI
class TestConsolidationE2E:
    def test_consolidate_psur_keytruda_6mo(self):
        """No injected client — FDAClient() constructs its own session, exercising the
        real construction path (requires both OPENFDA_API_KEY and OPENAI_API_KEY per
        initialize_config())."""
        result = consolidate_psur("KEYTRUDA", "6mo")

        assert isinstance(result, ConsolidationResult)
        assert result.dedup.total_output_count <= result.retrieval.total_retrieved_count
        assert result.formula_version == RANKING_FORMULA_VERSION
        assert isinstance(result.pharm_class, str)
        assert isinstance(result.class_counts_available, bool)
        assert result.ranking.total_signals == len(result.ranking.ranked_signals)

        # KEYTRUDA over a 6-month PSUR window reliably has hundreds of live FAERS
        # reports — these assertions prove real data actually flowed through the
        # full pipeline (identity -> retrieval -> dedup -> label -> ranking),
        # rather than merely checking types/structure that would also pass on an
        # empty result.
        assert result.retrieval.total_retrieved_count > 0
        assert len(result.ranking.ranked_signals) > 0

        first_signal = result.ranking.ranked_signals[0]
        assert first_signal.seriousness_tier in SERIOUSNESS_TIER_ORDER
        assert first_signal.strength_of_evidence_tier in STRENGTH_TIER_ORDER
        assert first_signal.reversibility_tier in REVERSIBILITY_TIER_ORDER
        assert first_signal.public_health_tier in PUBLIC_HEALTH_TIER_ORDER

        label_summary = result.ranking.label_summary
        assert label_summary.total_symptoms > 0
        assert (
            label_summary.labeled_count
            + label_summary.unlabeled_count
            + label_summary.unknown_count
            == label_summary.total_symptoms
        )
