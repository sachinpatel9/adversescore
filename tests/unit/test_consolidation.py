"""
Unit tests for consolidation.py (Phase 6) — pipeline wiring across drug identity
resolution, chunked FDA retrieval, deduplication, label lookup, and ranking.
All FDAClient methods and resolve_drug_identity are mocked/patched; no HTTP calls.
"""
from unittest.mock import MagicMock, patch

import pytest

from adverse_score.consolidation import consolidate_psur, ConsolidationResult, ConsolidationError
from adverse_score.drug_identity import DrugIdentity, DrugIdentityError
from adverse_score.deduplication import DedupResult
from adverse_score.ranking import RankingResult
from adverse_score.label_classifier import LabelClassificationResult
from adverse_score.fda_client import PSURRetrievalResult, ChunkResult


def _identity(**overrides):
    defaults = dict(
        canonical_name="KEYTRUDA",
        brand_names=["KEYTRUDA"],
        generic_names=["PEMBROLIZUMAB"],
        substance_names=["PEMBROLIZUMAB"],
        application_numbers=["BLA125514"],
        market_authorization_date=None,
        market_authorization_date_source="drugsfda.json:BLA125514",
        resolution_confidence="EXACT",
    )
    defaults.update(overrides)
    return DrugIdentity(**defaults)


def _chunk_result(reports=None, label="chunk-1-20240101-to-20240401"):
    reports = reports or []
    return ChunkResult(
        label=label, start_date="20240101", end_date="20240401",
        reports=reports, retrieved_count=len(reports),
        estimated_total_count=len(reports), truncated=False, error=None,
    )


def _retrieval_result(reports=None):
    reports = reports or []
    chunk = _chunk_result(reports=reports)
    return PSURRetrievalResult(
        canonical_query_variants=["KEYTRUDA"],
        period_start="20240101",
        period_end="20240401",
        chunks=[chunk],
        reports=reports,
        total_retrieved_count=len(reports),
        total_estimated_count=len(reports),
        any_chunk_truncated=False,
        used_fallback_anchor=False,
        fallback_reason=None,
    )


def _report(report_id, symptom_list=None, drug_names=None, date="20240101"):
    symptom_list = symptom_list or ["NAUSEA"]
    return {
        "report_id": report_id,
        "date": date,
        "severity": "Serious",
        "is_death": False,
        "is_hospitalization": False,
        "symptoms": ", ".join(symptom_list),
        "company": "PHARMA-001",
        "symptom_list": symptom_list,
        "drug_names": drug_names or ["KEYTRUDA"],
        "safetyreportversion": 1,
        "reactions": [{"term": s, "outcome_code": None} for s in symptom_list],
    }


def _dedup_result(reports):
    return DedupResult(
        reports=reports, total_input_count=len(reports), total_output_count=len(reports),
        removed_by_exact_id=0, removed_by_heuristic=0, audit_trail=[],
    )


def _make_mock_client():
    client = MagicMock()
    client.fetch_label_text.return_value = "known reactions include nausea"
    client._discover_drug_class.return_value = "PD-1/PDL-1 INHIBITOR"
    client._fetch_symptom_counts.return_value = {"NAUSEA": 500}
    return client


class TestHappyPathWiring:
    def test_all_stages_composed_correctly(self):
        identity = _identity()
        reports = [_report("R1"), _report("R2", symptom_list=["FATIGUE"])]
        retrieval = _retrieval_result(reports)
        client = _make_mock_client()
        client.fetch_psur_reports.return_value = retrieval

        with patch("adverse_score.consolidation.resolve_drug_identity", return_value=identity) as mock_resolve, \
             patch("adverse_score.consolidation.deduplicate_reports", wraps=lambda r: _dedup_result(r)) as mock_dedup, \
             patch("adverse_score.consolidation.rank_signals") as mock_rank:
            fake_ranking = MagicMock(spec=RankingResult)
            fake_ranking.formula_version = "1.0"
            fake_ranking.total_signals = 2
            mock_rank.return_value = fake_ranking

            result = consolidate_psur("KEYTRUDA", "6mo", client=client)

        assert isinstance(result, ConsolidationResult)
        assert result.drug_identity is identity
        assert result.period == "6mo"
        assert result.retrieval is retrieval
        assert result.dedup.reports == reports
        assert result.ranking is fake_ranking
        assert result.pharm_class == "PD-1/PDL-1 INHIBITOR"
        assert result.class_counts_available is True
        assert result.formula_version == "1.0"

        mock_resolve.assert_called_once_with("KEYTRUDA", client=client)
        client.fetch_psur_reports.assert_called_once_with(
            ["KEYTRUDA", "PEMBROLIZUMAB"], identity.market_authorization_date, "6mo")
        client.fetch_label_text.assert_called_once_with("KEYTRUDA")
        client._discover_drug_class.assert_called_once_with("KEYTRUDA")
        client._fetch_symptom_counts.assert_called_once_with(
            pharm_class="PD-1/PDL-1 INHIBITOR",
            start_date=retrieval.period_start, end_date=retrieval.period_end,
        )
        mock_dedup.assert_called_once_with(reports)
        mock_rank.assert_called_once()

    def test_label_summary_threaded_through_from_rank_signals(self):
        """rank_signals is mocked in the happy-path test above with
        MagicMock(spec=RankingResult), which auto-mocks .label_summary too — that
        alone doesn't prove consolidate_psur actually surfaces whatever
        LabelClassificationResult rank_signals produces. Build a real (non-mocked)
        LabelClassificationResult, attach it to the mocked RankingResult, and
        confirm it comes back out on result.ranking.label_summary unchanged."""
        identity = _identity()
        reports = [_report("R1"), _report("R2", symptom_list=["FATIGUE"])]
        retrieval = _retrieval_result(reports)
        client = _make_mock_client()
        client.fetch_psur_reports.return_value = retrieval

        real_label_summary = LabelClassificationResult(
            statuses={"NAUSEA": "LABELED", "FATIGUE": "UNLABELED"},
            total_symptoms=2,
            labeled_count=1,
            unlabeled_count=1,
            unknown_count=0,
        )

        with patch("adverse_score.consolidation.resolve_drug_identity", return_value=identity), \
             patch("adverse_score.consolidation.deduplicate_reports", wraps=lambda r: _dedup_result(r)), \
             patch("adverse_score.consolidation.rank_signals") as mock_rank:
            fake_ranking = MagicMock(spec=RankingResult)
            fake_ranking.formula_version = "1.0"
            fake_ranking.total_signals = 2
            fake_ranking.label_summary = real_label_summary
            mock_rank.return_value = fake_ranking

            result = consolidate_psur("KEYTRUDA", "6mo", client=client)

        assert isinstance(result, ConsolidationResult)
        assert result.ranking.label_summary is real_label_summary
        assert result.ranking.label_summary.total_symptoms == 2
        assert result.ranking.label_summary.labeled_count == 1
        assert result.ranking.label_summary.unlabeled_count == 1
        assert result.ranking.label_summary.unknown_count == 0


class TestIdentityResolutionFailure:
    def test_drug_identity_error_halts_pipeline(self):
        error = DrugIdentityError(
            query="NOTADRUG", reason="NOT_FOUND",
            message="No FDA-registered drug found matching 'NOTADRUG'.",
            attempted_variants=["label.json exact brand/generic: NOTADRUG"],
        )
        client = _make_mock_client()

        with patch("adverse_score.consolidation.resolve_drug_identity", return_value=error), \
             patch("adverse_score.consolidation.deduplicate_reports") as mock_dedup, \
             patch("adverse_score.consolidation.rank_signals") as mock_rank:
            result = consolidate_psur("NOTADRUG", "6mo", client=client)

        assert isinstance(result, ConsolidationError)
        assert result.stage == "IDENTITY_RESOLUTION"
        assert result.reason == "NOT_FOUND"
        assert result.drug_name == "NOTADRUG"
        assert result.message == error.message

        client.fetch_psur_reports.assert_not_called()
        mock_dedup.assert_not_called()
        mock_rank.assert_not_called()


class TestClientConstructionFailure:
    def test_missing_api_keys_returns_structured_error(self):
        with patch("adverse_score.consolidation.FDAClient", side_effect=EnvironmentError(
                "OPENFDA_API_KEY is not set in environment variables. Please set it in your .env file.")), \
             patch("adverse_score.consolidation.resolve_drug_identity") as mock_resolve:
            result = consolidate_psur("KEYTRUDA", "6mo")

        assert isinstance(result, ConsolidationError)
        assert result.stage == "CLIENT_CONSTRUCTION"
        assert result.reason == "MISSING_API_KEYS"
        assert result.drug_name == "KEYTRUDA"
        assert "OPENFDA_API_KEY" in result.message
        mock_resolve.assert_not_called()


class TestEmptyReportsAfterDedup:
    def test_zero_reports_still_returns_valid_result(self):
        identity = _identity()
        retrieval = _retrieval_result([])
        client = _make_mock_client()
        client.fetch_psur_reports.return_value = retrieval

        with patch("adverse_score.consolidation.resolve_drug_identity", return_value=identity):
            result = consolidate_psur("KEYTRUDA", "6mo", client=client)

        assert isinstance(result, ConsolidationResult)
        assert result.dedup.total_output_count == 0
        assert result.ranking.total_signals == 0
        assert result.ranking.ranked_signals == []


class TestPharmClassDiscoveryFails:
    def test_empty_pharm_class_still_calls_rank_signals_with_empty_class_counts(self):
        identity = _identity()
        reports = [_report("R1")]
        retrieval = _retrieval_result(reports)
        client = _make_mock_client()
        client.fetch_psur_reports.return_value = retrieval
        client._discover_drug_class.return_value = ""

        with patch("adverse_score.consolidation.resolve_drug_identity", return_value=identity), \
             patch("adverse_score.consolidation.rank_signals") as mock_rank:
            fake_ranking = MagicMock(spec=RankingResult)
            fake_ranking.formula_version = "1.0"
            fake_ranking.total_signals = 1
            mock_rank.return_value = fake_ranking

            result = consolidate_psur("KEYTRUDA", "6mo", client=client)

        assert isinstance(result, ConsolidationResult)
        assert result.pharm_class == ""
        assert result.class_counts_available is False
        client._fetch_symptom_counts.assert_not_called()
        mock_rank.assert_called_once()
        called_args = mock_rank.call_args[0]
        assert called_args[1] == {}

    def test_pharm_class_found_but_symptom_counts_empty_still_class_counts_unavailable(self):
        """class_counts_available must reflect whether class_counts is genuinely
        populated (bool(class_counts)), not merely whether pharm_class discovery
        succeeded — a pharm_class can resolve while _fetch_symptom_counts still
        returns {} (e.g. no symptom data in that date range)."""
        identity = _identity()
        reports = [_report("R1")]
        retrieval = _retrieval_result(reports)
        client = _make_mock_client()
        client.fetch_psur_reports.return_value = retrieval
        client._discover_drug_class.return_value = "PD-1/PDL-1 INHIBITOR"
        client._fetch_symptom_counts.return_value = {}

        with patch("adverse_score.consolidation.resolve_drug_identity", return_value=identity), \
             patch("adverse_score.consolidation.rank_signals") as mock_rank:
            fake_ranking = MagicMock(spec=RankingResult)
            fake_ranking.formula_version = "1.0"
            fake_ranking.total_signals = 1
            mock_rank.return_value = fake_ranking

            result = consolidate_psur("KEYTRUDA", "6mo", client=client)

        assert isinstance(result, ConsolidationResult)
        assert result.pharm_class == "PD-1/PDL-1 INHIBITOR"
        assert result.class_counts_available is False
        client._fetch_symptom_counts.assert_called_once_with(
            pharm_class="PD-1/PDL-1 INHIBITOR",
            start_date=retrieval.period_start, end_date=retrieval.period_end,
        )
        called_args = mock_rank.call_args[0]
        assert called_args[1] == {}


class TestNameVariantConstruction:
    def test_dedup_and_blank_filtering_across_brand_generic_substance(self):
        identity = _identity(
            canonical_name="KEYTRUDA",
            brand_names=["KEYTRUDA", ""],
            generic_names=["PEMBROLIZUMAB", "KEYTRUDA", "  "],
            substance_names=["PEMBROLIZUMAB", "PEMBROLIZUMAB-SUBSTANCE"],
        )
        reports = [_report("R1")]
        retrieval = _retrieval_result(reports)
        client = _make_mock_client()
        client.fetch_psur_reports.return_value = retrieval

        with patch("adverse_score.consolidation.resolve_drug_identity", return_value=identity):
            consolidate_psur("KEYTRUDA", "6mo", client=client)

        called_variants = client.fetch_psur_reports.call_args[0][0]
        assert called_variants == ["KEYTRUDA", "PEMBROLIZUMAB", "PEMBROLIZUMAB-SUBSTANCE"]

    def test_no_name_variants_returns_structured_retrieval_error(self):
        identity = _identity(brand_names=[], generic_names=[], substance_names=[])
        client = _make_mock_client()

        with patch("adverse_score.consolidation.resolve_drug_identity", return_value=identity):
            result = consolidate_psur("KEYTRUDA", "6mo", client=client)

        assert isinstance(result, ConsolidationError)
        assert result.stage == "RETRIEVAL"
        assert result.reason == "NO_NAME_VARIANTS"
        client.fetch_psur_reports.assert_not_called()
