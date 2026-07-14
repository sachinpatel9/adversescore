"""
Unit tests for ranking.py — the Phase 5 deterministic, lexicographic-tiered
signal ranking engine (Seriousness & Outcome, Strength of Evidence,
Reversibility, Public Health Impact). No HTTP calls; all inputs pre-fetched.
"""

import dataclasses
import pytest

from adverse_score.ranking import rank_signals, RankedSignal, RankingResult
from adverse_score.config import (
    RANKING_FORMULA_VERSION,
    PUBLIC_HEALTH_HIGH_VOLUME_THRESHOLD,
    PUBLIC_HEALTH_MODERATE_VOLUME_THRESHOLD,
)


def _report(report_id, symptom_list, reactions=None, severity="Non-Serious",
            is_death=False, is_hospitalization=False, drug_names=None, date="20250101"):
    """Builds a minimal flattened-report dict matching deduplication.py's DedupResult
    shape (symptom_list, reactions, severity, is_death, is_hospitalization, drug_names, date)."""
    return {
        "report_id": report_id,
        "date": date,
        "severity": severity,
        "is_death": is_death,
        "is_hospitalization": is_hospitalization,
        "symptoms": ", ".join(symptom_list),
        "company": "PHARMA-001",
        "symptom_list": symptom_list,
        "drug_names": drug_names or ["TESTDRUG"],
        "safetyreportversion": 1,
        "reactions": reactions if reactions is not None else [
            {"term": s, "outcome_code": None} for s in symptom_list
        ],
    }


def _big_class_counts(symptoms, base=1000):
    """Large background counts so PRR math doesn't blow up on tiny denominators."""
    return {s: base for s in symptoms}


class TestSeriousnessIsPrimarySortKey:
    """Seriousness must dominate the sort — a DEATH-tier signal ranks above a
    NON_SERIOUS-tier signal even when the NON_SERIOUS signal has stronger evidence."""

    def test_death_signal_ranks_above_non_serious_with_better_prr(self):
        # DEATH_SYMPTOM: few reports, weak PRR, but a death.
        # NON_SERIOUS_SYMPTOM: many reports, strong PRR, but never serious.
        reports = []
        for i in range(3):
            reports.append(_report(f"D{i}", ["DEATH_SYMPTOM"], is_death=True, severity="Serious"))
        for i in range(200):
            reports.append(_report(f"N{i}", ["NON_SERIOUS_SYMPTOM"], severity="Non-Serious"))

        class_counts = {"DEATH_SYMPTOM": 1000, "NON_SERIOUS_SYMPTOM": 210}

        result = rank_signals(reports, class_counts, label_text="")
        by_symptom = {s.symptom: s for s in result.ranked_signals}

        assert by_symptom["DEATH_SYMPTOM"].rank < by_symptom["NON_SERIOUS_SYMPTOM"].rank
        assert by_symptom["DEATH_SYMPTOM"].seriousness_tier == "DEATH"
        assert by_symptom["NON_SERIOUS_SYMPTOM"].seriousness_tier == "NON_SERIOUS"
        # Confirm the NON_SERIOUS signal really does have stronger/comparable evidence,
        # proving the primary sort key is seriousness, not a blended score.
        assert by_symptom["NON_SERIOUS_SYMPTOM"].prr_metrics["drug_cases"] > \
            by_symptom["DEATH_SYMPTOM"].prr_metrics["drug_cases"]

    def test_seriousness_tier_ignores_reports_not_containing_symptom(self):
        """Direct unit test of _seriousness_tier_for_symptom's own defensive
        per-report symptom filter (mirrors the reversibility-tier equivalent test in
        TestReversibilityTiering). rank_signals() always pre-filters via
        symptom_to_reports before calling this helper, so without this direct call
        the guard itself — 'a death report for a DIFFERENT symptom must never leak
        into this symptom's seriousness tier' — is never actually exercised."""
        from adverse_score.ranking import _seriousness_tier_for_symptom

        reports = [
            _report("R1", ["OTHER_SYMPTOM"], is_death=True, severity="Serious"),
            _report("R2", ["TARGET_SYMPTOM"], severity="Non-Serious"),
        ]
        tier = _seriousness_tier_for_symptom(reports, "TARGET_SYMPTOM")
        assert tier == "NON_SERIOUS"


class TestLabelStatusTieBreak:
    """Two signals identical in every other criterion: UNLABELED ranks above LABELED."""

    def test_unlabeled_ranks_above_labeled_at_equal_seriousness_and_volume(self):
        reports = []
        for i in range(50):
            reports.append(_report(f"U{i}", ["UNLABELED_SYMPTOM"], severity="Serious"))
        for i in range(50):
            reports.append(_report(f"L{i}", ["LABELED_SYMPTOM"], severity="Serious"))

        class_counts = {"UNLABELED_SYMPTOM": 1000, "LABELED_SYMPTOM": 1000}
        label_text = "known reactions include labeled_symptom in some patients"

        result = rank_signals(reports, class_counts, label_text=label_text)
        by_symptom = {s.symptom: s for s in result.ranked_signals}

        assert by_symptom["LABELED_SYMPTOM"].prr_metrics["label_status"] == "LABELED"
        assert by_symptom["UNLABELED_SYMPTOM"].prr_metrics["label_status"] == "UNLABELED"
        assert by_symptom["UNLABELED_SYMPTOM"].rank < by_symptom["LABELED_SYMPTOM"].rank

    def test_strong_unlabeled_ranks_above_strong_labeled(self):
        """Same tie-break inside the Strong (signal_detected=True) bucket: both signals
        have a genuinely detected PRR signal, only label status differs."""
        reports = []
        for i in range(50):
            reports.append(_report(f"U{i}", ["UNLABELED_SYMPTOM"], severity="Serious"))
        for i in range(50):
            reports.append(_report(f"L{i}", ["LABELED_SYMPTOM"], severity="Serious"))

        # Tiny background rates for both symptoms against a large background total
        # push PRR (and its CI lower bound) far above the signal threshold.
        class_counts = {"UNLABELED_SYMPTOM": 10, "LABELED_SYMPTOM": 10, "OTHER_BACKGROUND": 2000}
        label_text = "known reactions include labeled_symptom in some patients"

        result = rank_signals(reports, class_counts, label_text=label_text)
        by_symptom = {s.symptom: s for s in result.ranked_signals}

        assert by_symptom["UNLABELED_SYMPTOM"].prr_metrics["signal_detected"] is True
        assert by_symptom["LABELED_SYMPTOM"].prr_metrics["signal_detected"] is True
        assert by_symptom["UNLABELED_SYMPTOM"].strength_of_evidence_tier == "STRONG_UNLABELED"
        assert by_symptom["LABELED_SYMPTOM"].strength_of_evidence_tier == "STRONG_LABELED"
        assert by_symptom["UNLABELED_SYMPTOM"].rank < by_symptom["LABELED_SYMPTOM"].rank

    def test_five_tier_flat_strength_ordering(self):
        """UNKNOWN_LABEL_STATUS is its own single lowest-priority bucket — even when
        signal_detected is True — and the full 5-tier flat order is exactly as designed
        (not nested under Strong/Weak)."""
        from adverse_score.ranking import _strength_of_evidence_tier
        from adverse_score.config import STRENGTH_TIER_ORDER

        assert _strength_of_evidence_tier(True, "UNLABELED") == "STRONG_UNLABELED"
        assert _strength_of_evidence_tier(True, "LABELED") == "STRONG_LABELED"
        assert _strength_of_evidence_tier(False, "UNLABELED") == "WEAK_UNLABELED"
        assert _strength_of_evidence_tier(False, "LABELED") == "WEAK_LABELED"
        assert _strength_of_evidence_tier(True, "LABEL_STATUS_UNKNOWN") == "UNKNOWN_LABEL_STATUS"
        assert _strength_of_evidence_tier(False, "LABEL_STATUS_UNKNOWN") == "UNKNOWN_LABEL_STATUS"
        assert STRENGTH_TIER_ORDER == (
            "STRONG_UNLABELED", "STRONG_LABELED", "WEAK_UNLABELED",
            "WEAK_LABELED", "UNKNOWN_LABEL_STATUS",
        )


class TestFormulaVersion:
    def test_formula_version_present(self):
        reports = [_report("R1", ["NAUSEA"])]
        result = rank_signals(reports, {"NAUSEA": 100}, label_text="")
        assert result.formula_version == "1.0"
        assert result.formula_version == RANKING_FORMULA_VERSION


class TestReversibilityTiering:
    def test_fatal_outcome_anywhere_tiers_fatal(self):
        reports = [
            _report("R1", ["CARDIAC ARREST"],
                    reactions=[{"term": "CARDIAC ARREST", "outcome_code": 5}]),
            _report("R2", ["CARDIAC ARREST"],
                    reactions=[{"term": "CARDIAC ARREST", "outcome_code": 1}]),
        ]
        result = rank_signals(reports, {"CARDIAC ARREST": 500}, label_text="")
        signal = result.ranked_signals[0]
        assert signal.symptom == "CARDIAC ARREST"
        assert signal.reversibility_tier == "FATAL"

    def test_zero_non_none_outcome_codes_tiers_unknown_not_reversible(self):
        reports = [
            _report("R1", ["RASH"], reactions=[{"term": "RASH", "outcome_code": None}]),
            _report("R2", ["RASH"], reactions=[]),  # no reactions entries at all
        ]
        result = rank_signals(reports, {"RASH": 500}, label_text="")
        signal = result.ranked_signals[0]
        assert signal.symptom == "RASH"
        assert signal.reversibility_tier == "UNKNOWN"

    def test_recovered_code_tiers_reversible(self):
        reports = [
            _report("R1", ["NAUSEA"], reactions=[{"term": "NAUSEA", "outcome_code": 1}]),
        ]
        result = rank_signals(reports, {"NAUSEA": 500}, label_text="")
        assert result.ranked_signals[0].reversibility_tier == "REVERSIBLE"

    def test_not_recovered_code_tiers_poor(self):
        reports = [
            _report("R1", ["DISABILITY"], reactions=[{"term": "DISABILITY", "outcome_code": 3}]),
        ]
        result = rank_signals(reports, {"DISABILITY": 500}, label_text="")
        assert result.ranked_signals[0].reversibility_tier == "POOR"

    def test_reaction_terms_are_matched_per_symptom_not_globally(self):
        """A report with two symptoms (one fatal outcome, one recovered outcome) must
        attribute each outcome to its own matching symptom, not blend them."""
        reports = [
            _report("R1", ["SYMPTOM_A", "SYMPTOM_B"], reactions=[
                {"term": "SYMPTOM_A", "outcome_code": 5},
                {"term": "SYMPTOM_B", "outcome_code": 1},
            ]),
        ]
        result = rank_signals(reports, {"SYMPTOM_A": 500, "SYMPTOM_B": 500}, label_text="")
        by_symptom = {s.symptom: s for s in result.ranked_signals}
        assert by_symptom["SYMPTOM_A"].reversibility_tier == "FATAL"
        assert by_symptom["SYMPTOM_B"].reversibility_tier == "REVERSIBLE"

    def test_unknown_outcome_code_present_tiers_unknown_not_reversible(self):
        """A report that DOES supply a reactionoutcome code, but one that maps to
        FAERS's own 'Unknown' bucket (code 6, REACTION_OUTCOME_UNKNOWN_CODE), must
        still tier as UNKNOWN. This is a distinct code path from 'no code supplied at
        all' (tested above via outcome_code=None) — here codes_seen is non-empty, so
        the tiering must fall through _code_tier's final branch rather than the
        early-return on an empty codes_seen list."""
        reports = [
            _report("R1", ["MYSTERY_EVENT"], reactions=[{"term": "MYSTERY_EVENT", "outcome_code": 6}]),
        ]
        result = rank_signals(reports, {"MYSTERY_EVENT": 500}, label_text="")
        assert result.ranked_signals[0].reversibility_tier == "UNKNOWN"

    def test_reversibility_tier_ignores_reports_not_containing_symptom(self):
        """Direct unit test of _reversibility_tier_for_symptom's own defensive
        per-report symptom filter. rank_signals() always pre-filters the report list
        via symptom_to_reports before calling this helper, so this guard is otherwise
        never exercised by any rank_signals()-level test — but the helper is called
        directly here (as the module already does for _strength_of_evidence_tier
        above) to confirm the guard itself is correct: an unrelated report's FATAL
        outcome must never leak into a different symptom's reversibility tier."""
        from adverse_score.ranking import _reversibility_tier_for_symptom

        reports = [
            _report("R1", ["OTHER_SYMPTOM"], reactions=[{"term": "OTHER_SYMPTOM", "outcome_code": 5}]),
            _report("R2", ["TARGET_SYMPTOM"], reactions=[{"term": "TARGET_SYMPTOM", "outcome_code": 1}]),
        ]
        tier = _reversibility_tier_for_symptom(reports, "TARGET_SYMPTOM")
        assert tier == "REVERSIBLE"


class TestPublicHealthTiering:
    def test_at_and_above_high_threshold_is_high(self):
        n = PUBLIC_HEALTH_HIGH_VOLUME_THRESHOLD
        reports = [_report(f"R{i}", ["COMMON_SYMPTOM"]) for i in range(n)]
        result = rank_signals(reports, {"COMMON_SYMPTOM": n * 10}, label_text="")
        assert result.ranked_signals[0].public_health_tier == "HIGH"

    def test_above_high_threshold_is_high(self):
        n = PUBLIC_HEALTH_HIGH_VOLUME_THRESHOLD + 1
        reports = [_report(f"R{i}", ["COMMON_SYMPTOM"]) for i in range(n)]
        result = rank_signals(reports, {"COMMON_SYMPTOM": n * 10}, label_text="")
        assert result.ranked_signals[0].public_health_tier == "HIGH"

    def test_just_below_high_threshold_is_moderate(self):
        n = PUBLIC_HEALTH_HIGH_VOLUME_THRESHOLD - 1
        assert n >= PUBLIC_HEALTH_MODERATE_VOLUME_THRESHOLD
        reports = [_report(f"R{i}", ["MID_SYMPTOM"]) for i in range(n)]
        result = rank_signals(reports, {"MID_SYMPTOM": n * 10}, label_text="")
        assert result.ranked_signals[0].public_health_tier == "MODERATE"

    def test_at_moderate_threshold_is_moderate(self):
        n = PUBLIC_HEALTH_MODERATE_VOLUME_THRESHOLD
        reports = [_report(f"R{i}", ["EDGE_SYMPTOM"]) for i in range(n)]
        result = rank_signals(reports, {"EDGE_SYMPTOM": n * 10}, label_text="")
        assert result.ranked_signals[0].public_health_tier == "MODERATE"

    def test_below_moderate_threshold_is_low(self):
        n = PUBLIC_HEALTH_MODERATE_VOLUME_THRESHOLD - 1
        reports = [_report(f"R{i}", ["RARE_SYMPTOM"]) for i in range(n)]
        result = rank_signals(reports, {"RARE_SYMPTOM": n * 10}, label_text="")
        assert result.ranked_signals[0].public_health_tier == "LOW"


class TestEdgeCases:
    def test_empty_reports_list_returns_empty_result_no_exception(self):
        result = rank_signals([], {}, label_text="")
        assert isinstance(result, RankingResult)
        assert result.total_signals == 0
        assert result.ranked_signals == []
        assert result.formula_version == RANKING_FORMULA_VERSION

    def test_low_report_count_signal_still_present_not_dropped(self):
        """A signal below PRR_MINIMUM_DRUG_CASES must still be ranked, just likely
        lands in a weak-evidence tier — never excluded."""
        from adverse_score.config import PRR_MINIMUM_DRUG_CASES
        assert PRR_MINIMUM_DRUG_CASES > 1
        reports = [_report("R1", ["RARE_EVENT"])]  # exactly 1 mention, below the minimum
        result = rank_signals(reports, {"RARE_EVENT": 500}, label_text="")
        symptoms = [s.symptom for s in result.ranked_signals]
        assert "RARE_EVENT" in symptoms
        assert result.total_signals == 1
        signal = result.ranked_signals[0]
        assert signal.prr_metrics["signal_detected"] is False


class TestNoCompositeScalar:
    """Guardrail check: no attribute or dict key anywhere in RankedSignal/RankingResult
    is a single blended/composite numeric score."""

    def test_ranked_signal_fields_are_tiers_or_passthrough_only(self):
        reports = [_report("R1", ["NAUSEA"], is_death=True, severity="Serious",
                            reactions=[{"term": "NAUSEA", "outcome_code": 5}])]
        result = rank_signals(reports, {"NAUSEA": 500}, label_text="")
        signal = result.ranked_signals[0]

        field_names = {f.name for f in dataclasses.fields(signal)}
        assert field_names == {
            "symptom", "rank", "seriousness_tier", "strength_of_evidence_tier",
            "reversibility_tier", "public_health_tier", "prr_metrics", "report_count",
        }
        # The only numeric fields are rank (ordinal position, not a score) and
        # report_count (a raw count passthrough) plus whatever's inside prr_metrics
        # (pre-existing, untouched calculate_prr output — not ranking.py's own scalar).
        assert isinstance(signal.rank, int)
        assert isinstance(signal.report_count, int)
        for tier_field in ("seriousness_tier", "strength_of_evidence_tier",
                           "reversibility_tier", "public_health_tier"):
            assert isinstance(getattr(signal, tier_field), str)

    def test_ranking_result_fields_have_no_composite_score(self):
        result = rank_signals([], {}, label_text="")
        field_names = {f.name for f in dataclasses.fields(result)}
        assert field_names == {"ranked_signals", "label_summary", "formula_version", "total_signals"}


class TestDeterminism:
    def test_rank_signals_is_deterministic_across_runs(self):
        reports = [
            _report("R1", ["A_SYMPTOM"], is_death=True, severity="Serious"),
            _report("R2", ["B_SYMPTOM"], severity="Non-Serious"),
            _report("R3", ["C_SYMPTOM"], is_hospitalization=True, severity="Serious"),
        ]
        class_counts = {"A_SYMPTOM": 100, "B_SYMPTOM": 100, "C_SYMPTOM": 100}

        result1 = rank_signals(reports, class_counts, label_text="")
        result2 = rank_signals(reports, class_counts, label_text="")

        order1 = [s.symptom for s in result1.ranked_signals]
        order2 = [s.symptom for s in result2.ranked_signals]
        assert order1 == order2

    def test_ranks_are_1_indexed_and_sequential(self):
        reports = [
            _report("R1", ["X_SYMPTOM"]),
            _report("R2", ["Y_SYMPTOM"]),
            _report("R3", ["Z_SYMPTOM"]),
        ]
        class_counts = {"X_SYMPTOM": 100, "Y_SYMPTOM": 100, "Z_SYMPTOM": 100}
        result = rank_signals(reports, class_counts, label_text="")
        ranks = sorted(s.rank for s in result.ranked_signals)
        assert ranks == [1, 2, 3]


class TestLabelSummary:
    def test_label_summary_is_label_classification_result_with_aggregate_counts(self):
        reports = [
            _report("R1", ["NAUSEA", "HEADACHE"]),
        ]
        label_text = "adverse reactions include nausea"
        result = rank_signals(reports, {"NAUSEA": 100, "HEADACHE": 100}, label_text=label_text)
        assert result.label_summary.total_symptoms == 2
        assert result.label_summary.labeled_count == 1
        assert result.label_summary.unlabeled_count == 1
