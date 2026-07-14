"""
Unit tests for label_classifier.py — classify_label_status()
(LABELED/UNLABELED/LABEL_STATUS_UNKNOWN classification) and the batch
classify_label_statuses() / LabelClassificationResult (Phase 4).
"""

from adverse_score.label_classifier import (
    classify_label_status,
    classify_label_statuses,
    LabelClassificationResult,
)


class TestLabelClassification:
    """Tests for classify_label_status() — LABELED/UNLABELED/LABEL_STATUS_UNKNOWN classification."""

    def test_classify_labeled_when_symptom_in_label(self):
        """Symptom found in label text → LABELED."""
        label_text = "adverse reactions: nausea, fatigue, headache, hepatotoxicity"
        result = classify_label_status(label_text, "nausea, fatigue")
        assert result == "LABELED"

    def test_classify_unlabeled_when_symptom_not_in_label(self):
        """Symptom absent from label text → UNLABELED."""
        label_text = "adverse reactions: nausea, fatigue"
        result = classify_label_status(label_text, "pancreatitis")
        assert result == "UNLABELED"

    def test_classify_unknown_when_no_label_text(self):
        """Empty label text → LABEL_STATUS_UNKNOWN."""
        result = classify_label_status("", "nausea")
        assert result == "LABEL_STATUS_UNKNOWN"

    def test_classify_unknown_when_no_symptoms(self):
        """Empty symptoms string → LABEL_STATUS_UNKNOWN."""
        result = classify_label_status("adverse reactions: nausea", "")
        assert result == "LABEL_STATUS_UNKNOWN"


class TestLabelStatusesBatch:
    """Tests for classify_label_statuses() / LabelClassificationResult (Phase 4 batch API)."""

    def test_mixed_batch_labeled_and_unlabeled(self):
        """One symptom present in label text, one absent → both statuses present, counts match."""
        label_text = "adverse reactions: nausea, fatigue, headache"
        result = classify_label_statuses(label_text, ["nausea", "pancreatitis"])

        assert isinstance(result, LabelClassificationResult)
        assert result.statuses["NAUSEA"] == "LABELED"
        assert result.statuses["PANCREATITIS"] == "UNLABELED"
        assert result.total_symptoms == 2
        assert result.labeled_count == 1
        assert result.unlabeled_count == 1
        assert result.unknown_count == 0

    def test_duplicate_and_differently_cased_symptoms_collapse(self):
        """Differently-cased duplicates in the input list collapse to one entry."""
        label_text = "adverse reactions: nausea"
        result = classify_label_statuses(label_text, ["nausea", "Nausea", "NAUSEA"])

        assert result.statuses == {"NAUSEA": "LABELED"}
        assert result.total_symptoms == 1
        assert result.labeled_count == 1
        assert result.unlabeled_count == 0
        assert result.unknown_count == 0

    def test_empty_label_text_all_unknown(self):
        """Empty label_text → every symptom maps to LABEL_STATUS_UNKNOWN."""
        result = classify_label_statuses("", ["nausea", "fatigue", "headache"])

        assert result.total_symptoms == 3
        assert result.unknown_count == 3
        assert result.labeled_count == 0
        assert result.unlabeled_count == 0
        assert all(status == "LABEL_STATUS_UNKNOWN" for status in result.statuses.values())

    def test_empty_symptoms_list_no_crash(self):
        """Empty symptoms list → empty statuses dict, all counts zero, no exception."""
        result = classify_label_statuses("adverse reactions: nausea", [])

        assert result.statuses == {}
        assert result.total_symptoms == 0
        assert result.labeled_count == 0
        assert result.unlabeled_count == 0
        assert result.unknown_count == 0

    def test_whitespace_and_empty_entries_filtered_input_not_mutated(self):
        """Whitespace-only/empty entries are dropped, and the input list is not mutated."""
        symptoms = ["  ", "", "nausea", "\t"]
        original = list(symptoms)
        result = classify_label_statuses("adverse reactions: nausea", symptoms)

        assert result.statuses == {"NAUSEA": "LABELED"}
        assert result.total_symptoms == 1
        assert symptoms == original

    def test_all_whitespace_symptoms_list_yields_empty_result(self):
        """A symptoms list containing only whitespace entries behaves like an empty list."""
        result = classify_label_statuses("adverse reactions: nausea", ["  ", "\t", ""])

        assert result.statuses == {}
        assert result.total_symptoms == 0
        assert result.labeled_count == 0
        assert result.unlabeled_count == 0
        assert result.unknown_count == 0

    def test_counts_internally_consistent_on_mixed_batch(self):
        """labeled_count + unlabeled_count + unknown_count == total_symptoms on a mixed batch."""
        label_text = "adverse reactions: nausea, fatigue"
        result = classify_label_statuses(label_text, ["nausea", "fatigue", "pancreatitis"])

        assert result.labeled_count == 2
        assert result.unlabeled_count == 1
        assert result.unknown_count == 0
        assert (
            result.labeled_count + result.unlabeled_count + result.unknown_count
            == result.total_symptoms
        )
