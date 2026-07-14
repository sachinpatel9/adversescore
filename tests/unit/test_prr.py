"""
Unit tests for prr.py — pure PRR + Wald 95% CI math.

Tests call calculate_prr directly with pre-computed count dicts — no mocking needed.
"""

import math

from adverse_score.prr import calculate_prr


class TestPRR:
    """Tests for calculate_prr() — pure PRR + Wald 95% CI math."""

    def test_prr_division_by_zero_guard(self):
        """Returns prr=0.0 and signal_detected=False when denominators are zero."""
        result = calculate_prr({}, {}, "NAUSEA")
        assert result["prr"] == 0.0
        assert result["signal_detected"] is False

    def test_prr_insufficient_cases_guard(self):
        """Returns signal_detected=False when drug cases (a) < 3."""
        drug_counts = {"NAUSEA": 2, "HEADACHE": 10}
        class_counts = {"NAUSEA": 100, "HEADACHE": 500}
        result = calculate_prr(drug_counts, class_counts, "NAUSEA")
        assert result["signal_detected"] is False
        assert result["drug_cases"] == 2

    def test_prr_known_values(self):
        """PRR matches hand-computed value for a controlled 2x2 contingency table.

        Setup:
          a = drug + target symptom = 50
          a+b = total drug symptoms = 110
          c = class + target symptom = 500
          c+d = total class symptoms = 1220
        PRR = (a/(a+b)) / (c/(c+d)) = (50/110) / (500/1220) = 0.45454... / 0.40983... = 1.1090...
        """
        drug_counts = {"NAUSEA": 50, "FATIGUE": 30, "HEADACHE": 20, "HEPATOTOXICITY": 10}
        class_counts = {"NAUSEA": 500, "FATIGUE": 400, "HEADACHE": 300, "HEPATOTOXICITY": 20}
        result = calculate_prr(drug_counts, class_counts, "NAUSEA")

        a, a_plus_b = 50, 110
        c, c_plus_d = 500, 1220
        expected_prr = (a / a_plus_b) / (c / c_plus_d)
        assert result["prr"] == round(expected_prr, 2)
        assert result["drug_cases"] == 50
        assert result["class_cases"] == 500

    def test_prr_ci_lower_bound(self):
        """CI lower bound uses the Wald formula: exp(ln(PRR) - 1.96 * SE).

        SE = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
        """
        drug_counts = {"NAUSEA": 50, "FATIGUE": 30, "HEADACHE": 20, "HEPATOTOXICITY": 10}
        class_counts = {"NAUSEA": 500, "FATIGUE": 400, "HEADACHE": 300, "HEPATOTOXICITY": 20}
        result = calculate_prr(drug_counts, class_counts, "NAUSEA")

        a, a_plus_b = 50, 110
        c, c_plus_d = 500, 1220
        prr = (a / a_plus_b) / (c / c_plus_d)
        se = math.sqrt((1/a) + (1/c) - (1/a_plus_b) - (1/c_plus_d))
        expected_ci = math.exp(math.log(prr) - 1.96 * se)

        assert result["ci_lower"] == round(expected_ci, 2)

    def test_prr_strong_signal_detected(self):
        """When CI lower > 1.0 and a >= 3, signal_detected is True."""
        drug_counts = {"HEPATOTOXICITY": 100, "OTHER": 50}
        class_counts = {"HEPATOTOXICITY": 50, "OTHER": 5000}
        result = calculate_prr(drug_counts, class_counts, "HEPATOTOXICITY")

        # PRR = (100/150) / (50/5050) = 0.6667 / 0.0099 ≈ 67.3 → very strong signal
        assert result["signal_detected"] is True
        assert result["prr"] > 1.0
        assert result["ci_lower"] > 1.0

    def test_prr_class_zero_target_symptom(self):
        """When class has zero cases of the target symptom (c=0), guard returns prr=0.0."""
        drug_counts = {"NAUSEA": 10, "HEADACHE": 5}
        class_counts = {"HEADACHE": 100}
        result = calculate_prr(drug_counts, class_counts, "NAUSEA")
        assert result["prr"] == 0.0
        assert result["signal_detected"] is False
