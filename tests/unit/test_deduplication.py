"""
Unit tests for deduplication.py (Phase 3) — exact safetyreportid dedup with
version-aware tiebreak, and fallback heuristic dedup on drug_names/symptom_list/date.
"""

from adverse_score.deduplication import deduplicate_reports, DedupResult


def _report(report_id, version=1, drug_names=None, symptom_list=None, date="20250101", **overrides):
    """Builds a flattened-report-shaped dict (as produced by fda_client._flatten_results)."""
    entry = {
        "report_id": report_id,
        "date": date,
        "severity": "Serious",
        "is_death": False,
        "is_hospitalization": False,
        "symptoms": ", ".join(symptom_list or []),
        "company": "PHARMA-001",
        "symptom_list": symptom_list or [],
        "drug_names": drug_names or [],
        "safetyreportversion": version,
    }
    entry.update(overrides)
    return entry


class TestExactIdDedup:
    def test_higher_version_wins(self):
        r1 = _report("RPT-1", version=1, drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"])
        r2 = _report("RPT-1", version=2, drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"], company="AMENDED")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 1
        assert result.reports[0]["company"] == "AMENDED"
        assert result.removed_by_exact_id == 1
        assert result.removed_by_heuristic == 0
        exact_id_entries = [a for a in result.audit_trail if a["method"] == "exact_id"]
        assert len(exact_id_entries) == 1
        assert exact_id_entries[0]["kept_id"] == "RPT-1"

    def test_equal_or_missing_version_keeps_first_seen(self):
        r1 = _report("RPT-2", version=1, drug_names=["ASPIRIN"], symptom_list=["HEADACHE"], company="FIRST")
        r2 = _report("RPT-2", version=1, drug_names=["ASPIRIN"], symptom_list=["HEADACHE"], company="SECOND")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 1
        assert result.reports[0]["company"] == "FIRST"
        assert result.removed_by_exact_id == 1

    def test_missing_report_id_never_collapsed_in_exact_id_pass(self):
        """Multiple reports with report_id=None must not be treated as exact-ID
        duplicates of each other — a missing ID is 'no signal', not a match key."""
        r1 = _report(None, drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"], date="20250101")
        r2 = _report(None, drug_names=["OPDIVO"], symptom_list=["FATIGUE"], date="20250201")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 2
        assert result.removed_by_exact_id == 0


class TestHeuristicDedup:
    def test_identical_drugs_symptoms_date_merge(self):
        r1 = _report("RPT-A", drug_names=["KEYTRUDA"], symptom_list=["NAUSEA", "FATIGUE"], date="20250101")
        r2 = _report("RPT-B", drug_names=["KEYTRUDA"], symptom_list=["NAUSEA", "FATIGUE"], date="20250101")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 1
        assert result.removed_by_heuristic == 1
        heuristic_entries = [a for a in result.audit_trail if a["method"] == "heuristic"]
        assert len(heuristic_entries) == 1

    def test_normalization_is_case_and_whitespace_insensitive(self):
        r1 = _report("RPT-A", drug_names=["Keytruda"], symptom_list=[" Nausea ", "Fatigue"], date="20250101")
        r2 = _report("RPT-B", drug_names=["KEYTRUDA "], symptom_list=["NAUSEA", " FATIGUE"], date="20250101")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 1
        assert result.removed_by_heuristic == 1

    def test_partial_symptom_overlap_does_not_merge(self):
        """Highest-risk correctness property: symptom sets that partially overlap must NOT
        be treated as duplicates — full-set equality only."""
        r1 = _report("RPT-A", drug_names=["KEYTRUDA"], symptom_list=["NAUSEA", "FATIGUE"], date="20250101")
        r2 = _report("RPT-B", drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"], date="20250101")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 2
        assert result.removed_by_heuristic == 0

    def test_different_drug_names_does_not_merge(self):
        r1 = _report("RPT-A", drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"], date="20250101")
        r2 = _report("RPT-B", drug_names=["OPDIVO"], symptom_list=["NAUSEA"], date="20250101")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 2
        assert result.removed_by_heuristic == 0

    def test_different_date_does_not_merge(self):
        r1 = _report("RPT-A", drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"], date="20250101")
        r2 = _report("RPT-B", drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"], date="20250102")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 2
        assert result.removed_by_heuristic == 0

    def test_empty_drug_names_never_merged(self):
        r1 = _report("RPT-A", drug_names=[], symptom_list=["NAUSEA"], date="20250101")
        r2 = _report("RPT-B", drug_names=[], symptom_list=["NAUSEA"], date="20250101")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 2
        assert result.removed_by_heuristic == 0

    def test_empty_symptom_list_never_merged(self):
        r1 = _report("RPT-A", drug_names=["KEYTRUDA"], symptom_list=[], date="20250101")
        r2 = _report("RPT-B", drug_names=["KEYTRUDA"], symptom_list=[], date="20250101")
        result = deduplicate_reports([r1, r2])

        assert result.total_output_count == 2
        assert result.removed_by_heuristic == 0


class TestCombinedPipeline:
    def test_exact_and_heuristic_duplicates_together(self):
        # Exact-ID duplicate pair
        exact1 = _report("RPT-1", version=1, drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"], date="20250101")
        exact2 = _report("RPT-1", version=2, drug_names=["KEYTRUDA"], symptom_list=["NAUSEA"], date="20250101")
        # Heuristic-duplicate pair (different report_id, same drugs/symptoms/date)
        heur1 = _report("RPT-2", drug_names=["OPDIVO"], symptom_list=["FATIGUE"], date="20250201")
        heur2 = _report("RPT-3", drug_names=["OPDIVO"], symptom_list=["FATIGUE"], date="20250201")
        # Otherwise-unique report
        unique = _report("RPT-4", drug_names=["ASPIRIN"], symptom_list=["HEADACHE"], date="20250301")

        result = deduplicate_reports([exact1, exact2, heur1, heur2, unique])

        assert result.total_input_count == 5
        assert result.total_output_count == 3
        assert result.removed_by_exact_id == 1
        assert result.removed_by_heuristic == 1
        assert len(result.audit_trail) == 2

    def test_empty_input_list(self):
        result = deduplicate_reports([])

        assert isinstance(result, DedupResult)
        assert result.total_input_count == 0
        assert result.total_output_count == 0
        assert result.removed_by_exact_id == 0
        assert result.removed_by_heuristic == 0
        assert result.reports == []
        assert result.audit_trail == []
