"""
Unit tests for label_classifier.py — classify_label_status()
(LABELED/UNLABELED/LABEL_STATUS_UNKNOWN classification).
"""

from adverse_score.label_classifier import classify_label_status


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
