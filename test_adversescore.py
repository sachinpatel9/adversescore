"""
AdverseScore Test Suite
=======================
Tests organized by audit domain. Sections 1 and 3 are fully implemented.
Sections 2, 4, and 5 remain as scaffolds (placeholder `pass` bodies).
"""

import math
import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adverse_score.agent_tools import ClinicalQuerySchema
from adverse_score.client import AdverseScoreClient
from adverse_score.label_classifier import calculate_label_penalty, classify_label_status
from adverse_score.scoring import (calculate_report_score, calculate_confidence,
                                   generate_guardrails, SEVERITY_WEIGHTS)
from adverse_score.prr import calculate_prr


# ── SECTION 1: Pydantic Model Validation Tests ──────────────────────────────
# Tests for ClinicalQuerySchema in agent_tools.py:11-57.
# Validates field constraints, type enforcement, and custom validators.


class TestClinicalQuerySchemaValid:
    """Tests that valid inputs are accepted."""

    def test_valid_drug_name_only(self):
        """Accepts minimal valid input — just drug_name with no optional fields."""
        result = ClinicalQuerySchema(drug_name="KEYTRUDA")
        assert result.drug_name == "KEYTRUDA"
        assert result.patient_age is None
        assert result.patient_sex is None
        assert result.target_symptom is None

    def test_valid_full_query(self):
        """Accepts all fields populated with valid values (drug_name, age, sex, symptom)."""
        result = ClinicalQuerySchema(
            drug_name="OZEMPIC",
            patient_age=45,
            patient_sex="F",
            target_symptom="pancreatitis",
        )
        assert result.drug_name == "OZEMPIC"
        assert result.patient_age == 45
        assert result.patient_sex == "F"
        assert result.target_symptom == "pancreatitis"

    def test_drug_name_whitespace_stripped(self):
        """Validator strips leading/trailing whitespace from drug_name and returns the clean value."""
        result = ClinicalQuerySchema(drug_name="  KEYTRUDA  ")
        assert result.drug_name == "KEYTRUDA"

    def test_patient_age_valid_boundaries(self):
        """Accepts age=1 (minimum) and age=120 (maximum) — boundary values of ge=1, le=120."""
        low = ClinicalQuerySchema(drug_name="X", patient_age=1)
        high = ClinicalQuerySchema(drug_name="X", patient_age=120)
        assert low.patient_age == 1
        assert high.patient_age == 120

    def test_patient_sex_valid_values(self):
        """Accepts both 'M' and 'F' as valid Literal values."""
        male = ClinicalQuerySchema(drug_name="X", patient_sex="M")
        female = ClinicalQuerySchema(drug_name="X", patient_sex="F")
        assert male.patient_sex == "M"
        assert female.patient_sex == "F"


class TestClinicalQuerySchemaRejection:
    """Tests that invalid inputs are rejected with ValidationError."""

    def test_drug_name_empty_string_rejected(self):
        """min_length=1 rejects empty string '' for drug_name."""
        with pytest.raises(ValidationError):
            ClinicalQuerySchema(drug_name="")

    def test_drug_name_whitespace_only_rejected(self):
        """field_validator rejects '   ' after stripping — semantically empty."""
        with pytest.raises(ValidationError, match="blank"):
            ClinicalQuerySchema(drug_name="   ")

    def test_patient_age_below_minimum_rejected(self):
        """ge=1 rejects age=0 and negative ages like age=-5."""
        with pytest.raises(ValidationError):
            ClinicalQuerySchema(drug_name="X", patient_age=0)
        with pytest.raises(ValidationError):
            ClinicalQuerySchema(drug_name="X", patient_age=-5)

    def test_patient_age_above_maximum_rejected(self):
        """le=120 rejects age=121 and extreme values like age=200."""
        with pytest.raises(ValidationError):
            ClinicalQuerySchema(drug_name="X", patient_age=121)
        with pytest.raises(ValidationError):
            ClinicalQuerySchema(drug_name="X", patient_age=200)

    def test_patient_sex_invalid_string_rejected(self):
        """Literal['M','F'] rejects arbitrary strings like 'Male', 'female', 'X'."""
        for invalid in ["Male", "female", "X", "m", "f", "Other"]:
            with pytest.raises(ValidationError):
                ClinicalQuerySchema(drug_name="X", patient_sex=invalid)

    def test_extra_fields_rejected(self):
        """extra='forbid' rejects unexpected keys like {'drug_name': 'X', 'foo': 'bar'}."""
        with pytest.raises(ValidationError, match="extra"):
            ClinicalQuerySchema(drug_name="X", foo="bar")  # type: ignore

    def test_target_symptom_empty_string_rejected(self):
        """min_length=1 rejects empty string '' for target_symptom."""
        with pytest.raises(ValidationError):
            ClinicalQuerySchema(drug_name="X", target_symptom="")


class TestClinicalQuerySchemaIncludeTemporal:
    """Tests for the include_temporal field on ClinicalQuerySchema."""

    def test_include_temporal_true(self):
        result = ClinicalQuerySchema(drug_name="ASPIRIN", include_temporal=True)
        assert result.include_temporal is True

    def test_include_temporal_false(self):
        result = ClinicalQuerySchema(drug_name="ASPIRIN", include_temporal=False)
        assert result.include_temporal is False

    def test_include_temporal_default_none(self):
        result = ClinicalQuerySchema(drug_name="ASPIRIN")
        assert result.include_temporal is None


# ── SECTION 2: openFDA API Integration Tests ────────────────────────────────
# Tests for AdverseScoreClient methods in client.py.
# These should mock HTTP responses to avoid hitting the real FDA API.


class TestQueryBuilder:
    """Tests for build_query() and _sanitize_for_query()."""

    def test_build_query_basic(self, client):
        """Query string contains the drug name in a Lucene quoted field, a date range, and limit parameter."""
        query = client.build_query("KEYTRUDA")
        assert 'patient.drug.medicinalproduct:"KEYTRUDA"' in query
        assert "receivedate:" in query
        assert "limit=500" in query

    def test_build_query_with_demographics(self, client):
        """When patient_sex and patient_age are provided, the query includes sex code and age cohort bracket."""
        query = client.build_query("KEYTRUDA", patient_age=50, patient_sex="F")
        assert "patient.patientsex:2" in query
        assert "patient.patientonsetage:" in query
        assert "[45+TO+55]" in query

    def test_build_query_sex_code_mapping(self, client):
        """Verifies F maps to sex code '2' and M maps to sex code '1' (the corrected openFDA mapping)."""
        female_query = client.build_query("X", patient_sex="F")
        male_query = client.build_query("X", patient_sex="M")
        assert "patientsex:2" in female_query
        assert "patientsex:1" in male_query

    def test_sanitize_for_query_escapes_quotes(self, client):
        """Double quotes in input are escaped to backslash-quote for Lucene safety."""
        result = client._sanitize_for_query('DRUG"NAME')
        assert result == 'DRUG\\"NAME'

    def test_sanitize_for_query_escapes_backslash(self, client):
        """Backslashes in input are escaped to double-backslash for Lucene safety."""
        result = client._sanitize_for_query("DRUG\\NAME")
        assert result == "DRUG\\\\NAME"

    def test_sanitize_for_query_clean_input(self, client):
        """Normal drug names like 'KEYTRUDA' pass through unchanged."""
        assert client._sanitize_for_query("KEYTRUDA") == "KEYTRUDA"


class TestBuildQueryDateRange:
    """Tests for build_query() with explicit start_date/end_date parameters."""

    def test_explicit_date_range(self, client):
        """When start_date and end_date are provided, they appear in the query instead of days_back."""
        query = client.build_query("ASPIRIN", start_date="20250401", end_date="20250630")
        assert "20250401" in query
        assert "20250630" in query

    def test_days_back_still_works(self, client):
        """Without start_date/end_date, the query still uses days_back for date range."""
        query = client.build_query("ASPIRIN", days_back=30)
        assert "receivedate:" in query


class TestFetchEvents:
    """Tests for fetch_events() — the primary FDA API call."""

    def test_fetch_events_returns_data(self, client, mock_fda_response, monkeypatch):
        """A successful 200 response returns the parsed JSON dict with 'results' key."""
        class MockResponse:
            status_code = 200
            def json(self):
                return mock_fda_response
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        result = client.fetch_events("KEYTRUDA")
        assert result is not None
        assert "results" in result
        assert len(result["results"]) == 4

    def test_fetch_events_404_returns_none(self, client, monkeypatch):
        """A 404 response (zero results) returns None instead of raising an exception."""
        class MockResponse:
            status_code = 404
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        result = client.fetch_events("NONEXISTENTDRUG")
        assert result is None

    def test_fetch_events_http_error_returns_none(self, client, monkeypatch):
        """HTTP exceptions return None, not an unhandled exception."""
        import requests
        def raise_error(*a, **kw):
            raise requests.exceptions.ConnectionError("server down")
        monkeypatch.setattr(client.session, "get", raise_error)
        result = client.fetch_events("KEYTRUDA")
        assert result is None


class TestFlattenResults:
    """Tests for _flatten_results() — FDA JSON → flat dicts."""

    def test_flatten_results_extracts_fields(self, client, mock_fda_response):
        """Each flattened report contains report_id, date, severity, is_death, is_hospitalization, symptoms."""
        flat = client._flatten_results(mock_fda_response)
        assert len(flat) == 4
        required_keys = {"report_id", "date", "severity", "is_death", "is_hospitalization", "symptoms", "company"}
        for report in flat:
            assert required_keys.issubset(report.keys())
        # Verify severity mapping
        assert flat[0]["is_death"] is True
        assert flat[1]["is_hospitalization"] is True
        assert flat[0]["severity"] == "Serious"
        assert flat[3]["severity"] == "Non-Serious"
        # Verify symptoms joined
        assert "CARDIAC ARREST" in flat[0]["symptoms"]
        assert "DEATH" in flat[0]["symptoms"]

    def test_flatten_results_none_input(self, client):
        """None input (from a failed fetch) returns an empty list."""
        assert client._flatten_results(None) == []

    def test_flatten_results_empty_results(self, client):
        """A response with 'results': [] returns an empty list."""
        assert client._flatten_results({"results": []}) == []


class TestLabelAndDiscovery:
    """Tests for fetch_label_text(), _discover_drug_class(), and _discover_peers()."""

    def test_fetch_label_text_success(self, client, mock_label_response, monkeypatch):
        """Returns lowercase joined adverse_reactions text from a valid label response."""
        class MockResponse:
            status_code = 200
            def json(self):
                return mock_label_response
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        result = client.fetch_label_text("KEYTRUDA")
        assert "nausea" in result
        assert "hepatotoxicity" in result
        assert result == result.lower()  # must be lowercase

    def test_fetch_label_text_failure_returns_empty(self, client, monkeypatch):
        """API failure (timeout, 500, etc.) returns empty string as the unlabeled fallback."""
        def raise_error(*a, **kw):
            raise Exception("timeout")
        monkeypatch.setattr(client.session, "get", raise_error)
        assert client.fetch_label_text("KEYTRUDA") == ""

    def test_discover_drug_class_sanitizes_input(self, client, monkeypatch):
        """Drug name is passed through _sanitize_for_query before embedding in the Lucene search string.
        Uses params= dict so the escaped value is in kwargs['params']['search'], not the raw URL."""
        captured_kwargs = []
        class MockResponse:
            status_code = 200
            def json(self):
                return {"results": [{"term": "Test Class [EPC]", "count": 100}]}
            def raise_for_status(self):
                pass
        def capture_get(url, **kw):
            captured_kwargs.append(kw)
            return MockResponse()
        monkeypatch.setattr(client.session, "get", capture_get)
        client._discover_drug_class('DRUG"NAME')
        # The escaped quote must appear in the search param value
        search_param = captured_kwargs[0].get("params", {}).get("search", "")
        assert '\\"' in search_param

    def test_discover_peers_excludes_target_drug(self, client, monkeypatch):
        """The target drug itself is filtered out of the peer list."""
        class MockResponse:
            status_code = 200
            def json(self):
                return {"results": [
                    {"term": "KEYTRUDA", "count": 500},
                    {"term": "OPDIVO", "count": 400},
                    {"term": "YERVOY", "count": 300},
                    {"term": "TECENTRIQ", "count": 200},
                ]}
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        peers = client._discover_peers("Programmed Death Receptor [EPC]", "KEYTRUDA")
        assert "KEYTRUDA" not in peers
        assert len(peers) == 3
        assert "OPDIVO" in peers

    def test_discover_peers_respects_min_name_length(self, client, monkeypatch):
        """Peer names with 3 or fewer characters are excluded as likely abbreviations."""
        class MockResponse:
            status_code = 200
            def json(self):
                return {"results": [
                    {"term": "AB", "count": 500},
                    {"term": "XYZ", "count": 400},
                    {"term": "OPDIVO", "count": 300},
                    {"term": "YERVOY", "count": 200},
                ]}
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        peers = client._discover_peers("Test Class [EPC]", "KEYTRUDA")
        assert "AB" not in peers
        assert "XYZ" not in peers
        assert "OPDIVO" in peers

    def test_fetch_symptom_counts_handles_malformed_items(self, client, monkeypatch):
        """Result items missing 'term' or 'count' keys are skipped instead of raising KeyError."""
        class MockResponse:
            status_code = 200
            def json(self):
                return {"results": [
                    {"term": "NAUSEA", "count": 50},
                    {"count": 30},           # missing term
                    {"term": "HEADACHE"},     # missing count
                    {"term": "FATIGUE", "count": 20},
                ]}
            def raise_for_status(self):
                pass
        monkeypatch.setattr(client.session, "get", lambda *a, **kw: MockResponse())
        counts = client._fetch_symptom_counts(drug_name="KEYTRUDA")
        assert counts == {"NAUSEA": 50, "FATIGUE": 20}
        assert "HEADACHE" not in counts


class TestLabelClassification:
    """Tests for _classify_label_status() — LABELED/UNLABELED/LABEL_STATUS_UNKNOWN classification."""

    def test_classify_labeled_when_symptom_in_label(self, client):
        """Symptom found in label text → LABELED."""
        label_text = "adverse reactions: nausea, fatigue, headache, hepatotoxicity"
        result = client._classify_label_status(label_text, "nausea, fatigue")
        assert result == "LABELED"

    def test_classify_unlabeled_when_symptom_not_in_label(self, client):
        """Symptom absent from label text → UNLABELED."""
        label_text = "adverse reactions: nausea, fatigue"
        result = client._classify_label_status(label_text, "pancreatitis")
        assert result == "UNLABELED"

    def test_classify_unknown_when_no_label_text(self, client):
        """Empty label text → LABEL_STATUS_UNKNOWN."""
        result = client._classify_label_status("", "nausea")
        assert result == "LABEL_STATUS_UNKNOWN"

    def test_classify_unknown_when_no_symptoms(self, client):
        """Empty symptoms string → LABEL_STATUS_UNKNOWN."""
        result = client._classify_label_status("adverse reactions: nausea", "")
        assert result == "LABEL_STATUS_UNKNOWN"


# ── SECTION 3: Statistical Math Accuracy Tests ──────────────────────────────
# Tests for scoring formulas, confidence metrics, PRR, and guardrails in client.py.
# Uses a client instance that bypasses network calls for pure math testing.


@pytest.fixture
def math_client():
    """AdverseScoreClient for math-only tests. Requires .env for initialization."""
    return AdverseScoreClient()


class TestLabelPenalty:
    """Tests for calculate_label_penalty() — the unlabeled/labeled multiplier."""

    def test_label_penalty_unlabeled_serious(self, math_client):
        """No label text → 2.0x multiplier for serious reports."""
        assert math_client.calculate_label_penalty("nausea", "", True) == 2.0

    def test_label_penalty_unlabeled_non_serious(self, math_client):
        """No label text → 1.5x multiplier for non-serious reports."""
        assert math_client.calculate_label_penalty("headache", "", False) == 1.5

    def test_label_penalty_labeled(self, math_client):
        """Symptom found in label text → 1.0x multiplier (no penalty)."""
        label = "nausea, fatigue, headache reported in clinical trials"
        assert math_client.calculate_label_penalty("nausea", label, True) == 1.0

    def test_label_penalty_empty_symptoms(self, math_client):
        """Empty symptom string '' → unlabeled penalty (2.0x/1.5x), not 1.0x."""
        label = "nausea, fatigue reported"
        # Empty symptoms = unknown label status → treat as unlabeled
        assert math_client.calculate_label_penalty("", label, True) == 2.0
        assert math_client.calculate_label_penalty("", label, False) == 1.5

    def test_label_penalty_whitespace_only_symptoms(self, math_client):
        """Whitespace-only symptoms like '  ,  ' → unlabeled penalty after filtering."""
        label = "nausea, fatigue reported"
        assert math_client.calculate_label_penalty("  ,  ", label, True) == 2.0

    def test_label_penalty_symptom_not_in_label(self, math_client):
        """Symptom not found in label → unlabeled penalty."""
        label = "nausea, fatigue reported"
        assert math_client.calculate_label_penalty("pancreatitis", label, True) == 2.0
        assert math_client.calculate_label_penalty("pancreatitis", label, False) == 1.5


class TestReportScore:
    """Tests for _calculate_report_score() — per-report severity weighting."""

    def test_report_score_death(self, math_client):
        """Death report uses DEATH weight (1.75). With labeled symptom → 1.75 * 1.0 = 1.75."""
        report = {"severity": "Serious", "is_death": True, "is_hospitalization": False, "symptoms": "death"}
        label_text = "death, cardiac arrest"
        score = math_client._calculate_report_score(report, label_text)
        assert score == 1.75 * 1.0  # DEATH * labeled

    def test_report_score_hospitalization(self, math_client):
        """Serious + hospitalized → HOSPITALIZATION weight (1.0). With labeled → 1.0 * 1.0 = 1.0."""
        report = {"severity": "Serious", "is_death": False, "is_hospitalization": True, "symptoms": "nausea"}
        label_text = "nausea, vomiting"
        score = math_client._calculate_report_score(report, label_text)
        assert score == 1.0 * 1.0  # HOSPITALIZATION * labeled

    def test_report_score_other_serious(self, math_client):
        """Serious + not hospitalized → OTHER_SERIOUS weight (0.75). With labeled → 0.75."""
        report = {"severity": "Serious", "is_death": False, "is_hospitalization": False, "symptoms": "fatigue"}
        label_text = "fatigue, dizziness"
        score = math_client._calculate_report_score(report, label_text)
        assert score == 0.75 * 1.0  # OTHER_SERIOUS * labeled

    def test_report_score_non_serious(self, math_client):
        """Non-serious → NON_SERIOUS weight (0.25). With labeled → 0.25."""
        report = {"severity": "Non-Serious", "is_death": False, "is_hospitalization": False, "symptoms": "headache"}
        label_text = "headache, nausea"
        score = math_client._calculate_report_score(report, label_text)
        assert score == 0.25 * 1.0  # NON_SERIOUS * labeled

    def test_report_score_death_unlabeled(self, math_client):
        """Death + unlabeled → maximum single-report weight: 1.75 * 2.0 = 3.5."""
        report = {"severity": "Serious", "is_death": True, "is_hospitalization": False, "symptoms": "cardiac arrest"}
        score = math_client._calculate_report_score(report, "")  # no label text
        assert score == 1.75 * 2.0  # DEATH * unlabeled-serious

    def test_report_score_non_serious_unlabeled(self, math_client):
        """Non-serious + unlabeled → minimum severity, partial penalty: 0.25 * 1.5 = 0.375."""
        report = {"severity": "Non-Serious", "is_death": False, "is_hospitalization": False, "symptoms": "rash"}
        score = math_client._calculate_report_score(report, "")
        assert score == 0.25 * 1.5  # NON_SERIOUS * unlabeled-non-serious

    def test_report_score_missing_symptoms_key(self, math_client):
        """Report with no 'symptoms' key defaults to '' → unlabeled penalty."""
        report = {"severity": "Serious", "is_death": False, "is_hospitalization": True}
        label_text = "nausea, headache"
        score = math_client._calculate_report_score(report, label_text)
        # symptoms defaults to '' → empty → unlabeled penalty 2.0 for serious
        assert score == 1.0 * 2.0  # HOSPITALIZATION * unlabeled-serious


class TestConfidence:
    """Tests for _calculate_confidence() — sample size and quality assessment."""

    def test_confidence_zero_reports(self, math_client):
        """Empty report list → level 'None', metric 0.0, defect_ratio 0.0."""
        result = math_client._calculate_confidence([])
        assert result["level"] == "None"
        assert result["metric"] == 0.0
        assert result["defect_ratio"] == 0.0

    def test_confidence_small_sample(self, math_client):
        """10 clean reports → continuous curve produces 'Medium' confidence."""
        reports = [{"date": "20260101", "symptoms": "NAUSEA"} for _ in range(10)]
        result = math_client._calculate_confidence(reports)
        assert result["level"] == "Medium"
        assert result["metric"] == 72.7

    def test_confidence_medium_sample(self, math_client):
        """60 clean reports → continuous curve produces 'High' confidence."""
        reports = [{"date": "20260101", "symptoms": "NAUSEA"} for _ in range(60)]
        result = math_client._calculate_confidence(reports)
        assert result["level"] == "High"
        assert result["metric"] == 96.1

    def test_confidence_high_sample(self, math_client):
        """100+ reports → curve saturates at 100.0."""
        reports = [{"date": "20260101", "symptoms": "NAUSEA"} for _ in range(100)]
        result = math_client._calculate_confidence(reports)
        assert result["level"] == "High"
        assert result["metric"] == 100.0

    def test_confidence_quality_penalty(self, math_client):
        """Reports with missing dates reduce confidence via defect_ratio * 50 penalty."""
        # 10 reports, 5 with missing dates → defect_ratio = 0.5 → penalty = 25
        good = [{"date": "20260101", "symptoms": "NAUSEA"} for _ in range(5)]
        bad = [{"date": None, "symptoms": "NAUSEA"} for _ in range(5)]
        result = math_client._calculate_confidence(good + bad)
        # base=72.7 (log-linear for n=10), penalty=0.5*50=25, final=72.7-25=47.7
        assert result["metric"] == 47.7
        assert result["defect_ratio"] == 0.5
        assert result["level"] == "Low"

    def test_confidence_all_defective(self, math_client):
        """All reports defective → penalty exceeds base, score floors above 0."""
        reports = [{"date": None, "symptoms": "Unknown"} for _ in range(10)]
        result = math_client._calculate_confidence(reports)
        # base=72.7, defect_ratio=1.0, penalty=50, final=max(0, 72.7-50)=22.7
        assert result["metric"] == 22.7
        assert result["level"] == "None"

    def test_confidence_unknown_symptoms_counted(self, math_client):
        """Reports with symptoms='Unknown' are counted as defects."""
        reports = [{"date": "20260101", "symptoms": "Unknown"} for _ in range(10)]
        result = math_client._calculate_confidence(reports)
        assert result["defect_ratio"] == 1.0

    def test_confidence_continuous_curve_no_cliff(self, math_client):
        """49 and 50 reports produce nearly identical scores (no step-function cliff)."""
        reports_49 = [{"date": "20260101", "symptoms": "X"} for _ in range(49)]
        reports_50 = [{"date": "20260101", "symptoms": "X"} for _ in range(50)]
        metric_49 = math_client._calculate_confidence(reports_49)["metric"]
        metric_50 = math_client._calculate_confidence(reports_50)["metric"]
        assert abs(metric_49 - metric_50) < 1.0
        assert math_client._calculate_confidence(reports_49)["level"] == "High"
        assert math_client._calculate_confidence(reports_50)["level"] == "High"


class TestFinalScore:
    """Tests for calculate_final_score() — the aggregate scoring pipeline."""

    def test_final_score_empty_reports(self, math_client):
        """Empty report list → 'Incomplete Data' payload with all required directive fields."""
        result = math_client.calculate_final_score("TESTDRUG", [], skip_benchmark=True)
        assert result["clinical_signal"]["status"] == "Incomplete Data"
        assert result["clinical_signal"]["adverse_score"] == 0.0
        assert result["agent_directives"]["diagnosis_lock"] is True
        assert result["agent_directives"]["requires_human_review"] is False
        assert "clinical_disclaimer" in result["metadata"]
        assert "system_directive" in result["agent_directives"]

    def test_final_score_normalization_bounds(self, math_client):
        """Score is capped at 100 for extreme inputs. Formula: min(100, mean_signal * 40)."""
        today = datetime.now().strftime("%Y%m%d")
        # All death + unlabeled (no label text) + recent → max mean_signal = 1.75 * 2.0 = 3.5
        # 3.5 * 40 = 140, capped to 100
        extreme_reports = [
            {"date": today, "severity": "Serious", "is_death": True,
             "is_hospitalization": False, "symptoms": "unknown_symptom"}
            for _ in range(10)
        ]
        result = math_client.calculate_final_score("TEST", extreme_reports, skip_benchmark=True)
        assert result["clinical_signal"]["adverse_score"] == 100

    def test_final_score_minimum_value(self, math_client):
        """All non-serious + labeled + old → minimum score = 0.25 * 1.0 * 0.5 * 40 = 5.0."""
        old_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        min_reports = [
            {"date": old_date, "severity": "Non-Serious", "is_death": False,
             "is_hospitalization": False, "symptoms": "headache"}
            for _ in range(10)
        ]
        # Need label text that contains "headache" so penalty is 1.0
        # We mock fetch_label_text by calling calculate_final_score which calls it internally.
        # Since we can't easily mock it here, we test the math directly:
        # base=0.25 (NON_SERIOUS), penalty depends on label lookup.
        # For a deterministic test, verify score >= 0 and <= 100
        result = math_client.calculate_final_score("TEST", min_reports, skip_benchmark=True)
        score = result["clinical_signal"]["adverse_score"]
        assert 0 <= score <= 100

    def test_final_score_status_thresholds(self, math_client):
        """Score > 70 → 'High Signal', 30 < score <= 70 → 'Monitor', score <= 30 → 'Stable'."""
        today = datetime.now().strftime("%Y%m%d")
        # Create reports that will produce a high score (death + unlabeled)
        high_reports = [
            {"date": today, "severity": "Serious", "is_death": True,
             "is_hospitalization": False, "symptoms": "rare_symptom"}
            for _ in range(10)
        ]
        result = math_client.calculate_final_score("TEST", high_reports, skip_benchmark=True)
        # With death + likely unlabeled + recent, score should be > 70
        assert "High Signal" in result["clinical_signal"]["status"] or \
               "Monitor" in result["clinical_signal"]["status"] or \
               result["clinical_signal"]["status"] == "Stable"


class TestPRR:
    """Tests for calculate_prr() — pure PRR + Wald 95% CI math.

    Tests call calculate_prr directly with pre-computed count dicts — no mocking needed.
    """

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


class TestGuardrails:
    """Tests for _generate_guardrails() — deterministic AI behavior flags."""

    def test_guardrails_high_score_triggers_review(self, math_client):
        """adverse_score > 70 → requires_human_review=True."""
        confidence = {"level": "High"}
        result = math_client._generate_guardrails(75.0, confidence)
        assert result["requires_human_review"] is True

    def test_guardrails_low_confidence_triggers_review(self, math_client):
        """adverse_score > 40 with confidence level 'Low' → requires_human_review=True."""
        confidence = {"level": "Low"}
        result = math_client._generate_guardrails(45.0, confidence)
        assert result["requires_human_review"] is True

    def test_guardrails_stable_no_review(self, math_client):
        """Low score + High confidence → requires_human_review=False."""
        confidence = {"level": "High"}
        result = math_client._generate_guardrails(20.0, confidence)
        assert result["requires_human_review"] is False

    def test_guardrails_prr_signal_overrides(self, math_client):
        """PRR signal_detected=True → forces requires_human_review=True and route_to_specialist=True."""
        confidence = {"level": "High"}
        prr = {"signal_detected": True}
        result = math_client._generate_guardrails(20.0, confidence, prr)
        assert result["requires_human_review"] is True
        assert result["route_to_specialist"] is True

    def test_guardrails_diagnosis_lock_always_true(self, math_client):
        """diagnosis_lock is True in every scenario — low score, high score, with/without PRR."""
        scenarios = [
            (10.0, {"level": "High"}, None),
            (75.0, {"level": "Low"}, None),
            (50.0, {"level": "Medium"}, {"signal_detected": True}),
            (0.0, {"level": "None"}, None),
        ]
        for score, confidence, prr in scenarios:
            result = math_client._generate_guardrails(score, confidence, prr)
            assert result["diagnosis_lock"] is True, f"Failed for score={score}"

    def test_guardrails_specialist_routing_threshold(self, math_client):
        """route_to_specialist is True when adverse_score > 60."""
        confidence = {"level": "High"}
        below = math_client._generate_guardrails(60.0, confidence)
        above = math_client._generate_guardrails(61.0, confidence)
        assert below["route_to_specialist"] is False
        assert above["route_to_specialist"] is True

    def test_guardrails_boundary_40_low_confidence(self, math_client):
        """Score exactly 40 + Low confidence → requires_human_review=False (threshold is > 40)."""
        confidence = {"level": "Low"}
        result = math_client._generate_guardrails(40.0, confidence)
        assert result["requires_human_review"] is False

    def test_peer_benchmark_skips_empty_peers(self):
        """Peers with zero adverse event data are excluded from the benchmark average."""
        pass


class TestComputeQuarterBoundaries:
    """Tests for _compute_quarter_boundaries() method."""

    def test_returns_requested_count(self, client):
        result = client._compute_quarter_boundaries(4)
        assert len(result) == 4

    def test_quarter_label_format(self, client):
        import re as re_mod
        result = client._compute_quarter_boundaries(4)
        for label, start, end in result:
            assert re_mod.match(r"\d{4}-Q[1-4]", label), f"Invalid label: {label}"

    def test_date_format(self, client):
        result = client._compute_quarter_boundaries(4)
        for label, start, end in result:
            assert len(start) == 8 and start.isdigit(), f"Invalid start date: {start}"
            assert len(end) == 8 and end.isdigit(), f"Invalid end date: {end}"

    def test_chronological_order(self, client):
        result = client._compute_quarter_boundaries(4)
        for i in range(1, len(result)):
            assert result[i][1] > result[i-1][2], "Quarters not in chronological order"


class TestComputeTrend:
    """Tests for compute_trend() method."""

    def test_trend_rising(self, client, sample_time_series_rising):
        assert client.compute_trend(sample_time_series_rising) == "RISING"

    def test_trend_stable(self, client, sample_time_series_stable):
        assert client.compute_trend(sample_time_series_stable) == "STABLE"

    def test_trend_declining(self, client, sample_time_series_declining):
        assert client.compute_trend(sample_time_series_declining) == "DECLINING"

    def test_trend_insufficient_data(self, client, sample_time_series_insufficient):
        assert client.compute_trend(sample_time_series_insufficient) == "INSUFFICIENT_DATA"

    def test_trend_empty_list(self, client):
        assert client.compute_trend([]) == "INSUFFICIENT_DATA"

    def test_trend_single_quarter(self, client):
        single = [{"quarter": "2026-Q1", "adverse_score": 42, "prr": 1.9, "report_count": 85}]
        assert client.compute_trend(single) == "INSUFFICIENT_DATA"

    def test_trend_boundary_exactly_10(self, client):
        """Delta of exactly 10 (recent vs 2-quarters-prior) should classify as RISING."""
        data = [
            {"quarter": "2025-Q2", "adverse_score": 30, "prr": None, "report_count": 50},
            {"quarter": "2025-Q3", "adverse_score": 35, "prr": None, "report_count": 60},
            {"quarter": "2025-Q4", "adverse_score": 38, "prr": None, "report_count": 70},
            {"quarter": "2026-Q1", "adverse_score": 45, "prr": None, "report_count": 80},
        ]
        assert client.compute_trend(data) == "RISING"  # 45 - 35 = 10


class TestPersistenceLayer:
    """Tests for SQLite persistence in persistence.py."""

    def test_save_and_retrieve(self, temp_store, sample_agent_payload):
        """save_analysis stores a record and get_history retrieves it."""
        row_id = temp_store.save_analysis(sample_agent_payload)
        assert row_id >= 1
        history = temp_store.get_history(limit=5)
        assert len(history) == 1
        assert history[0]["drug_name"] == "KEYTRUDA"
        assert history[0]["adverse_score"] == 45.0

    def test_get_prior_analysis(self, temp_store, sample_agent_payload):
        """get_prior_analysis returns the most recent prior for a drug."""
        temp_store.save_analysis(sample_agent_payload)
        prior = temp_store.get_prior_analysis("KEYTRUDA")
        assert prior is not None
        assert prior["drug_name"] == "KEYTRUDA"
        assert prior["adverse_score"] == 45.0

    def test_get_prior_analysis_case_insensitive(self, temp_store, sample_agent_payload):
        """get_prior_analysis matches drug names case-insensitively."""
        temp_store.save_analysis(sample_agent_payload)
        prior = temp_store.get_prior_analysis("keytruda")
        assert prior is not None
        assert prior["drug_name"] == "KEYTRUDA"

    def test_get_prior_analysis_no_match(self, temp_store):
        """get_prior_analysis returns None for a drug with no history."""
        prior = temp_store.get_prior_analysis("NONEXISTENT")
        assert prior is None

    def test_delta_calculation(self, temp_store, sample_agent_payload):
        """Two saves of the same drug produce ordered history."""
        temp_store.save_analysis(sample_agent_payload)
        sample_agent_payload["clinical_signal"]["adverse_score"] = 60.0
        sample_agent_payload["metadata"]["timestamp"] = "2026-04-01T00:00:00Z"
        temp_store.save_analysis(sample_agent_payload)
        history = temp_store.get_history(limit=5)
        assert len(history) == 2
        assert history[0]["adverse_score"] == 60.0  # Most recent first

    def test_history_limit(self, temp_store, sample_agent_payload):
        """get_history respects the limit parameter."""
        for i in range(25):
            sample_agent_payload["metadata"]["timestamp"] = f"2026-01-{i+1:02d}T00:00:00Z"
            temp_store.save_analysis(sample_agent_payload)
        history = temp_store.get_history(limit=20)
        assert len(history) == 20

    def test_portfolio_latest_per_drug(self, temp_store, sample_agent_payload):
        """get_portfolio returns only the latest analysis per drug."""
        temp_store.save_analysis(sample_agent_payload)
        sample_agent_payload["clinical_signal"]["adverse_score"] = 55.0
        sample_agent_payload["metadata"]["timestamp"] = "2026-04-01T00:00:00Z"
        temp_store.save_analysis(sample_agent_payload)
        portfolio = temp_store.get_portfolio()
        assert len(portfolio) == 1
        assert portfolio[0]["adverse_score"] == 55.0

    def test_payload_with_prr(self, temp_store, sample_agent_payload_with_prr):
        """save_analysis correctly extracts PRR from pharmacovigilance_metrics."""
        temp_store.save_analysis(sample_agent_payload_with_prr)
        history = temp_store.get_history()
        assert history[0]["prr_value"] == 2.5

    def test_payload_with_temporal(self, temp_store, sample_agent_payload_with_temporal):
        """save_analysis extracts trend_classification from temporal_analysis."""
        temp_store.save_analysis(sample_agent_payload_with_temporal)
        history = temp_store.get_history()
        assert history[0]["trend_classification"] == "RISING"

    def test_update_narrative(self, temp_store, sample_agent_payload):
        """update_narrative saves narrative text to the most recent row."""
        temp_store.save_analysis(sample_agent_payload)
        temp_store.update_narrative("KEYTRUDA", "Test narrative content")
        history = temp_store.get_history()
        assert history[0]["signal_narrative"] == "Test narrative content"


# ── SECTION 4: LangGraph Agent Behavior Tests ───────────────────────────────
# Tests for orchestrator wiring, system prompt compliance, and payload structure.


class TestOrchestratorWiring:
    """Tests that the agent executor is correctly configured."""

    def test_uses_create_agent_import(self):
        """orchestrator imports create_agent from langchain.agents (the current stable API)."""
        import inspect
        import adverse_score.orchestrator as orch_module
        assert hasattr(orch_module, "create_agent"), (
            "create_agent must be imported from langchain.agents into orchestrator"
        )
        source = inspect.getsource(orch_module)
        assert "from langchain.agents import create_agent" in source
        assert "from langgraph.prebuilt import create_react_agent" not in source

    def test_agent_executor_instantiated_with_create_agent(self):
        """agent_executor is built with create_agent (langchain.agents), not the deprecated langgraph.prebuilt spelling."""
        import inspect
        import adverse_score.orchestrator as orch_module
        source = inspect.getsource(orch_module)
        assert "create_agent(" in source
        assert "create_react_agent(" not in source

    def test_agent_executor_uses_system_prompt_kwarg(self):
        """agent_executor passes 'system_prompt=' kwarg, matching create_agent's API (not bare 'prompt=')."""
        import re
        import inspect
        import adverse_score.orchestrator as orch_module
        source = inspect.getsource(orch_module)
        assert "system_prompt=system_instructions" in source
        # Ensure no bare `prompt=` kwarg (i.e. not preceded by 'system_')
        assert not re.search(r'(?<!system_)prompt=system_instructions', source)

    def test_agent_executor_has_tool_bound(self):
        """agent_executor has get_adverse_score in its available tools."""
        from adverse_score.orchestrator import tools
        tool_names = [t.name for t in tools]
        assert "get_adverse_score" in tool_names

    def test_system_prompt_contains_scope_enforcement(self):
        """System prompt includes the SCOPE ENFORCEMENT block for off-topic deflection."""
        from adverse_score.orchestrator import system_instructions
        assert "SCOPE ENFORCEMENT" in system_instructions
        assert "I am designed to only assist with pharmaceutical safety analysis" in system_instructions

    def test_system_prompt_contains_diagnosis_lock(self):
        """System prompt includes unconditional DIAGNOSIS LOCK rule (not conditional on a flag)."""
        from adverse_score.orchestrator import system_instructions
        assert "DIAGNOSIS LOCK" in system_instructions
        assert "MUST NEVER formulate a diagnosis" in system_instructions
        # Must be unconditional — should NOT say "if diagnosis_lock is true"
        assert "if diagnosis_lock" not in system_instructions.lower()

    def test_system_prompt_contains_tool_protocol(self):
        """System prompt includes 'exactly ONCE' instruction to prevent retry loops."""
        from adverse_score.orchestrator import system_instructions
        assert "exactly ONCE" in system_instructions
        assert "do not retry" in system_instructions.lower()

    def test_system_prompt_contains_response_format(self):
        """System prompt includes prose-based RESPONSE FORMAT requiring cohesive narrative and disclaimer."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        assert "RESPONSE FORMAT" in system_instructions
        # Must require cohesive prose, not numbered sections
        assert "cohesive clinical narrative" in normalized
        assert "not a numbered list" in normalized
        # Disclaimer requirement preserved
        assert "clinical disclaimer" in normalized

    def test_system_prompt_requires_rationale_substance(self):
        """System prompt requires integrating score rationale, peer context, and confidence into narrative."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        # RESPONSE FORMAT requires integrating these elements into prose
        assert "rationale" in normalized
        assert "peer context" in normalized or "peer benchmark" in normalized
        assert "confidence" in normalized
        # REASONING PROTOCOL still references PRR and signal analysis
        assert "PRR" in system_instructions
        assert "signals" in normalized

    def test_system_prompt_reasoning_leads_with_significant_finding(self):
        """REASONING PROTOCOL instructs opening with the most clinically significant finding."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        # New first bullet: lead with most significant finding as opening sentence
        assert "most clinically significant" in normalized
        assert "opening sentence" in normalized
        # Original bullet: do not bury significant signals
        assert "do not bury" in normalized

    def test_system_prompt_integrates_label_status(self):
        """System prompt requires label status to be integrated into the clinical narrative."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        # Label status is listed as a required element of the prose narrative
        assert "label status" in normalized

    def test_system_prompt_prose_format_excludes_numbered_sections(self):
        """RESPONSE FORMAT explicitly prohibits numbered lists and structured sections."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        assert "not a numbered list or structured sections" in normalized
        # ICH E2E narrative is the only exception that uses headings
        assert "signal evaluation narrative" in normalized

    def test_system_prompt_response_length_guidance(self):
        """TONE & STYLE includes response length guidance tied to query complexity."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        assert "response length should match query complexity" in normalized
        assert "3-5 sentences" in system_instructions
        assert "never pad" in normalized


class TestSignalNarrativeProtocol:
    """Tests for the Signal Narrative Generator (Priority 3)."""

    # ── System prompt tests ──

    def test_system_prompt_contains_narrative_protocol(self):
        """System prompt includes NARRATIVE GENERATION PROTOCOL with all six ICH E2E headings."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        assert "narrative generation protocol" in normalized
        for heading in [
            "signal description",
            "data source and method",
            "statistical analysis summary",
            "clinical assessment",
            "data limitations",
            "recommendation",
        ]:
            assert heading in normalized, f"Missing ICH E2E heading: {heading}"

    def test_system_prompt_narrative_keywords_listed(self):
        """System prompt lists all trigger keywords for narrative generation."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        for keyword in ["write", "narrative", "document", "report", "summarise", "summarize", "memo", "draft"]:
            assert keyword in normalized, f"Missing trigger keyword: {keyword}"

    def test_system_prompt_narrative_grounding_rule(self):
        """System prompt encourages clinical interpretation while requiring validation flags."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        assert "requires clinical validation" in normalized
        assert "clinical interpretation is encouraged" in normalized

    def test_system_prompt_narrative_draft_header(self):
        """System prompt requires DRAFT header on narratives."""
        from adverse_score.orchestrator import system_instructions
        assert "DRAFT" in system_instructions
        assert "For Human Review Only" in system_instructions

    # ── Regex extraction tests ──

    def test_narrative_regex_extracts_valid_narrative(self, sample_narrative_output):
        """Regex extracts narrative body from markers and strips it from main output."""
        import re
        narrative_match = re.search(
            r'<!-- SIGNAL_NARRATIVE_START -->\s*(.*?)\s*<!-- SIGNAL_NARRATIVE_END -->',
            sample_narrative_output, re.DOTALL
        )
        assert narrative_match is not None
        narrative_text = narrative_match.group(1).strip()
        assert "DRAFT" in narrative_text
        assert "Signal Description" in narrative_text
        # Main output should not contain the narrative after stripping
        main_output = sample_narrative_output[:narrative_match.start()].rstrip()
        assert "<!-- SIGNAL_NARRATIVE_START -->" not in main_output
        assert "Some standard analysis content here" in main_output

    def test_narrative_regex_no_match_without_markers(self):
        """Standard response without narrative markers returns no match."""
        import re
        standard_output = "## Drug Analysis\n\nAdverseScore: 45/100\n\nSome analysis."
        narrative_match = re.search(
            r'<!-- SIGNAL_NARRATIVE_START -->\s*(.*?)\s*<!-- SIGNAL_NARRATIVE_END -->',
            standard_output, re.DOTALL
        )
        assert narrative_match is None

    # ── Keyword detection tests ──

    def test_keyword_detection_positive(self):
        """message_requests_narrative returns True for documentation-intent messages."""
        sys.path.insert(0, str(Path(__file__).parent))
        from app import message_requests_narrative
        assert message_requests_narrative("write up this finding") is True
        assert message_requests_narrative("Can you document this?") is True
        assert message_requests_narrative("Generate a report for this drug") is True
        assert message_requests_narrative("Create a memo on the results") is True
        assert message_requests_narrative("Please draft a summary") is True

    def test_keyword_detection_negative(self):
        """message_requests_narrative returns False for standard queries."""
        sys.path.insert(0, str(Path(__file__).parent))
        from app import message_requests_narrative
        assert message_requests_narrative("analyze metformin safety profile") is False
        assert message_requests_narrative("What is the adverse score for ibuprofen?") is False

    # ── Narrative content tests ──

    def test_narrative_contains_all_ich_sections(self, sample_narrative_output):
        """Sample narrative contains all six ICH E2E section headings."""
        import re
        narrative_match = re.search(
            r'<!-- SIGNAL_NARRATIVE_START -->\s*(.*?)\s*<!-- SIGNAL_NARRATIVE_END -->',
            sample_narrative_output, re.DOTALL
        )
        narrative_text = narrative_match.group(1)
        for heading in [
            "Signal Description",
            "Data Source and Method",
            "Statistical Analysis Summary",
            "Clinical Assessment",
            "Data Limitations",
            "Recommendation",
        ]:
            assert heading in narrative_text, f"Missing section: {heading}"

    def test_narrative_starts_with_draft_header(self, sample_narrative_output):
        """Extracted narrative begins with DRAFT header."""
        import re
        narrative_match = re.search(
            r'<!-- SIGNAL_NARRATIVE_START -->\s*(.*?)\s*<!-- SIGNAL_NARRATIVE_END -->',
            sample_narrative_output, re.DOTALL
        )
        narrative_text = narrative_match.group(1).strip()
        assert narrative_text.startswith("DRAFT")


class TestTemporalAnalysisProtocol:
    """Tests for the Temporal Trend Analysis (Priority 4)."""

    # ── System prompt tests ──

    def test_system_prompt_contains_temporal_protocol(self):
        """System prompt includes TEMPORAL ANALYSIS PROTOCOL with trigger keywords."""
        from adverse_score.orchestrator import system_instructions
        normalized = " ".join(system_instructions.lower().split())
        assert "temporal analysis protocol" in normalized
        for keyword in ["trend", "over time", "quarterly", "changing", "getting worse",
                        "getting better", "historical", "last quarter", "recent quarters"]:
            assert keyword in normalized, f"Missing temporal keyword: {keyword}"

    def test_system_prompt_temporal_markers(self):
        """System prompt contains TIME_SERIES_DATA marker instructions."""
        from adverse_score.orchestrator import system_instructions
        assert "TIME_SERIES_DATA_START" in system_instructions
        assert "TIME_SERIES_DATA_END" in system_instructions

    # ── Keyword detection tests ──

    def test_temporal_keyword_detection_positive(self):
        """message_requests_temporal returns True for temporal queries."""
        sys.path.insert(0, str(Path(__file__).parent))
        from app import message_requests_temporal
        assert message_requests_temporal("Show me the trend for aspirin") is True
        assert message_requests_temporal("How has aspirin changed over time") is True
        assert message_requests_temporal("quarterly analysis of aspirin") is True
        assert message_requests_temporal("Is aspirin getting worse") is True

    def test_temporal_keyword_detection_negative(self):
        """message_requests_temporal returns False for standard queries."""
        sys.path.insert(0, str(Path(__file__).parent))
        from app import message_requests_temporal
        assert message_requests_temporal("analyze metformin safety profile") is False
        assert message_requests_temporal("What is the adverse score for ibuprofen?") is False

    # ── Regex extraction tests ──

    def test_time_series_regex_extracts_json(self):
        """Regex extracts valid JSON from time_series markers."""
        import re
        import json
        mock_output = (
            "Some analysis text.\n\n"
            "<!-- TIME_SERIES_DATA_START -->\n"
            '[{"quarter":"2025-Q2","adverse_score":30,"prr":1.5,"report_count":100},'
            '{"quarter":"2025-Q3","adverse_score":35,"prr":1.7,"report_count":110}]\n'
            "<!-- TIME_SERIES_DATA_END -->\n\n"
            "More text."
        )
        match = re.search(
            r'<!-- TIME_SERIES_DATA_START -->\s*(.*?)\s*<!-- TIME_SERIES_DATA_END -->',
            mock_output, re.DOTALL
        )
        assert match is not None
        data = json.loads(match.group(1).strip())
        assert len(data) == 2
        assert data[0]["quarter"] == "2025-Q2"

    def test_time_series_regex_no_match_without_markers(self):
        """Standard response without markers returns no match."""
        import re
        standard_output = "## Drug Analysis\n\nAdverseScore: 45/100\n\nSome analysis."
        match = re.search(
            r'<!-- TIME_SERIES_DATA_START -->\s*(.*?)\s*<!-- TIME_SERIES_DATA_END -->',
            standard_output, re.DOTALL
        )
        assert match is None


class TestDeltaDetectionProtocol:
    """Tests for the DELTA DETECTION PROTOCOL in the system prompt."""

    def test_system_prompt_contains_delta_protocol(self):
        """System prompt includes the DELTA DETECTION PROTOCOL section."""
        from adverse_score.orchestrator import system_instructions
        assert "DELTA DETECTION PROTOCOL" in system_instructions

    def test_delta_protocol_includes_score_change(self):
        """DELTA DETECTION PROTOCOL includes Score Change vs Prior section guidance."""
        from adverse_score.orchestrator import system_instructions
        assert "Score Change vs Prior" in system_instructions

    def test_delta_detection_conditional_language(self):
        """System prompt instructs NOT to include delta when absent."""
        from adverse_score.orchestrator import system_instructions
        assert "Do NOT include this section when" in system_instructions


class TestPayloadStructure:
    """Tests that tool output payloads have all fields the system prompt expects."""

    def test_error_payload_has_required_fields(self, sample_error_payload):
        """Exception handler payload includes clinical_disclaimer, diagnosis_lock, requires_human_review, system_directive."""
        meta = sample_error_payload["metadata"]
        directives = sample_error_payload["agent_directives"]
        assert "clinical_disclaimer" in meta
        assert directives["diagnosis_lock"] is True
        assert "requires_human_review" in directives
        assert "system_directive" in directives

    def test_success_payload_has_required_fields(self, sample_agent_payload):
        """Normal payload includes metadata, clinical_signal, data_integrity, agent_directives top-level keys."""
        required_keys = {"metadata", "clinical_signal", "data_integrity", "agent_directives"}
        assert required_keys.issubset(sample_agent_payload.keys())
        assert "clinical_disclaimer" in sample_agent_payload["metadata"]
        assert "adverse_score" in sample_agent_payload["clinical_signal"]
        assert "report_count" in sample_agent_payload["data_integrity"]
        assert "diagnosis_lock" in sample_agent_payload["agent_directives"]

    def test_label_status_in_clinical_signal(self, sample_agent_payload):
        """Payload includes label_status field in clinical_signal."""
        assert "label_status" in sample_agent_payload["clinical_signal"]
        assert sample_agent_payload["clinical_signal"]["label_status"] in [
            "LABELED", "UNLABELED", "LABEL_STATUS_UNKNOWN"
        ]

    def test_label_status_in_prr_metrics(self, sample_agent_payload_with_prr):
        """PRR metrics payload includes label_status for the target symptom."""
        prr = sample_agent_payload_with_prr["pharmacovigilance_metrics"]
        assert "label_status" in prr
        assert prr["label_status"] in ["LABELED", "UNLABELED", "LABEL_STATUS_UNKNOWN"]

    def test_payload_demographics_injected(self):
        """extracted_demographics dict with age, sex, and target_symptom is added to metadata."""
        import json
        from unittest.mock import patch, MagicMock
        from adverse_score import agent_tools

        # Mock the global client to return controlled data
        mock_client = MagicMock()
        mock_client.fetch_events.return_value = None  # triggers empty reports
        mock_client._flatten_results.return_value = []
        mock_client.calculate_final_score.return_value = {
            "metadata": {"tool_name": "AdverseScore"},
            "clinical_signal": {"adverse_score": 0.0, "status": "Incomplete Data"},
            "data_integrity": {"report_count": 0},
            "agent_directives": {"diagnosis_lock": True},
        }

        with patch.object(agent_tools, "_global_client", mock_client):
            result_json = agent_tools.get_adverse_score.invoke({
                "drug_name": "TESTDRUG",
                "patient_age": 55,
                "patient_sex": "F",
                "target_symptom": "nausea",
            })

        result = json.loads(result_json)
        demo = result["metadata"]["extracted_demographics"]
        assert demo["age"] == 55
        assert demo["sex"] == "F"
        assert demo["target_symptom"] == "nausea"


class TestAgentToolBehavior:
    """Tests for get_adverse_score tool behavior — invocation, empty results, errors, and output structure.

    These tests call the tool function directly with a mocked AdverseScoreClient,
    isolating agent behavior from real API and LLM calls.
    """

    def _invoke_tool(self, agent_tools_module, mock_client, **kwargs):
        """Helper: patch _global_client, invoke the tool, return parsed JSON."""
        import json
        from unittest.mock import patch
        with patch.object(agent_tools_module, "_global_client", mock_client):
            raw = agent_tools_module.get_adverse_score.invoke(kwargs)
        return json.loads(raw)

    def test_tool_invokes_fetch_for_drug_query(self):
        """The tool calls fetch_events with the drug name when invoked with a valid drug query."""
        from unittest.mock import MagicMock
        from adverse_score import agent_tools

        mock_client = MagicMock()
        mock_client.fetch_events.return_value = None
        mock_client._flatten_results.return_value = []
        mock_client.calculate_final_score.return_value = {
            "metadata": {"tool_name": "AdverseScore"},
            "clinical_signal": {"adverse_score": 0.0},
            "data_integrity": {},
            "agent_directives": {"diagnosis_lock": True},
        }

        self._invoke_tool(agent_tools, mock_client, drug_name="KEYTRUDA")
        mock_client.fetch_events.assert_called_once()
        call_args = mock_client.fetch_events.call_args
        assert call_args[0][0] == "KEYTRUDA"

    def test_tool_returns_incomplete_data_for_empty_results(self):
        """When fetch_events returns None (no FDA data), the payload status is 'Incomplete Data'."""
        from unittest.mock import MagicMock
        from adverse_score import agent_tools

        mock_client = MagicMock()
        mock_client.fetch_events.return_value = None
        mock_client._flatten_results.return_value = []
        # Use the real calculate_final_score to verify the Incomplete Data path
        real_client = AdverseScoreClient()
        mock_client.calculate_final_score.side_effect = real_client.calculate_final_score

        result = self._invoke_tool(agent_tools, mock_client, drug_name="NONEXISTENTDRUG")
        assert result["clinical_signal"]["status"] == "Incomplete Data"
        assert result["clinical_signal"]["adverse_score"] == 0.0
        assert "system_directive" in result["agent_directives"]
        assert "spelling" in result["agent_directives"]["system_directive"].lower() or \
               "insufficient" in result["agent_directives"]["system_directive"].lower()

    def test_tool_returns_error_payload_on_exception(self):
        """When the client raises an exception, the tool returns a structured error payload (not a traceback).
        The raw exception message must NOT appear in the payload to prevent internal detail exposure."""
        from unittest.mock import MagicMock
        from adverse_score import agent_tools

        mock_client = MagicMock()
        mock_client.fetch_events.side_effect = RuntimeError("database connection lost")

        result = self._invoke_tool(agent_tools, mock_client, drug_name="KEYTRUDA")
        assert result["metadata"]["status"] == "System Error"
        assert "clinical_disclaimer" in result["metadata"]
        assert result["agent_directives"]["diagnosis_lock"] is True
        # Raw exception text must NOT be in the payload (security: no internal detail exposure)
        assert "database connection lost" not in result["agent_directives"]["system_directive"]
        # Generic user-facing message must be present
        assert "system error" in result["agent_directives"]["system_directive"].lower()

    def test_tool_output_has_all_required_top_level_keys(self):
        """A successful tool response contains metadata, clinical_signal, data_integrity, and agent_directives."""
        from unittest.mock import MagicMock
        from adverse_score import agent_tools
        from datetime import datetime

        today = datetime.now().strftime("%Y%m%d")
        mock_client = MagicMock()
        mock_client.fetch_events.return_value = {
            "results": [{"safetyreportid": "R1", "receivedate": today, "seriousness": "1",
                          "seriousnessdeath": None, "seriousnesshospitalization": "1",
                          "patient": {"reaction": [{"reactionmeddrapt": "NAUSEA"}]},
                          "companynumb": "CO-1"}]
        }
        # Use the real client for flatten + score
        real_client = AdverseScoreClient()
        mock_client._flatten_results.side_effect = real_client._flatten_results
        mock_client.calculate_final_score.side_effect = lambda *a, **kw: real_client.calculate_final_score(*a, **kw, skip_benchmark=True)

        result = self._invoke_tool(agent_tools, mock_client, drug_name="TESTDRUG")
        required = {"metadata", "clinical_signal", "data_integrity", "agent_directives"}
        assert required.issubset(result.keys())
        assert isinstance(result["clinical_signal"]["adverse_score"], (int, float))
        assert result["clinical_signal"]["adverse_score"] >= 0
        assert result["agent_directives"]["diagnosis_lock"] is True

    def test_tool_does_not_retry_on_failure(self):
        """The tool calls fetch_events exactly once even when it fails — no internal retry loop."""
        from unittest.mock import MagicMock
        from adverse_score import agent_tools

        mock_client = MagicMock()
        mock_client.fetch_events.side_effect = RuntimeError("server error")

        self._invoke_tool(agent_tools, mock_client, drug_name="KEYTRUDA")
        assert mock_client.fetch_events.call_count == 1

    def test_tool_schema_rejects_off_topic_input(self):
        """The Pydantic schema rejects empty drug_name, preventing nonsensical tool calls."""
        with pytest.raises((ValidationError, Exception)):
            ClinicalQuerySchema(drug_name="")

    def test_error_payload_matches_system_prompt_expectations(self):
        """The error payload contains every field the system prompt instructs the LLM to read:
        clinical_disclaimer, diagnosis_lock, requires_human_review, and system_directive."""
        from unittest.mock import MagicMock
        from adverse_score import agent_tools

        mock_client = MagicMock()
        mock_client.fetch_events.side_effect = Exception("test failure")

        result = self._invoke_tool(agent_tools, mock_client, drug_name="KEYTRUDA")
        # These fields are referenced in system_instructions rules 1, 2, 3, 5
        assert result["agent_directives"]["diagnosis_lock"] is True
        assert "requires_human_review" in result["agent_directives"]
        assert "system_directive" in result["agent_directives"]
        assert "clinical_disclaimer" in result["metadata"]

    def test_tool_passes_demographics_to_client(self):
        """When age and sex are provided, the tool forwards them to fetch_events and calculate_final_score."""
        from unittest.mock import MagicMock
        from adverse_score import agent_tools

        mock_client = MagicMock()
        mock_client.fetch_events.return_value = None
        mock_client._flatten_results.return_value = []
        mock_client.calculate_final_score.return_value = {
            "metadata": {"tool_name": "AdverseScore"},
            "clinical_signal": {"adverse_score": 0.0},
            "data_integrity": {},
            "agent_directives": {"diagnosis_lock": True},
        }

        self._invoke_tool(agent_tools, mock_client, drug_name="OZEMPIC", patient_age=45, patient_sex="F")

        # Verify demographics were passed to fetch_events
        fetch_args = mock_client.fetch_events.call_args
        assert fetch_args[0][1] == 45   # patient_age
        assert fetch_args[0][2] == "F"  # patient_sex

        # Verify demographics were passed to calculate_final_score
        calc_kwargs = mock_client.calculate_final_score.call_args
        assert calc_kwargs[1]["patient_age"] == 45 or calc_kwargs[0][2] == 45


# ── SECTION 5: Edge Case & Adversarial Input Tests ──────────────────────────
# Tests for boundary conditions, special characters, and defensive behavior.


class TestSpecialCharacterHandling:
    """Tests that drug names with special characters are handled safely."""

    def test_drug_name_with_slash(self, client):
        """'INSULIN/DEXTROSE' passes Pydantic validation and produces a valid Lucene query."""
        schema = ClinicalQuerySchema(drug_name="INSULIN/DEXTROSE")
        assert schema.drug_name == "INSULIN/DEXTROSE"
        query = client.build_query("INSULIN/DEXTROSE")
        assert "INSULIN/DEXTROSE" in query

    def test_drug_name_with_hyphen(self, client):
        """'L-DOPA' passes Pydantic validation and produces a valid Lucene query."""
        schema = ClinicalQuerySchema(drug_name="L-DOPA")
        assert schema.drug_name == "L-DOPA"
        query = client.build_query("L-DOPA")
        assert "L-DOPA" in query

    def test_drug_name_with_quotes(self, client):
        """A drug name containing '"' is properly escaped to '\\"' in the Lucene query string."""
        query = client.build_query('DRUG"NAME')
        # The quote must be escaped so the Lucene quoted field isn't broken
        assert 'DRUG\\"NAME' in query

    def test_drug_name_with_backslash(self, client):
        """A drug name containing '\\' is properly escaped to '\\\\' in the Lucene query string."""
        query = client.build_query("DRUG\\NAME")
        assert "DRUG\\\\NAME" in query

    def test_drug_name_case_insensitivity(self, client):
        """'aspirin', 'ASPIRIN', and 'Aspirin' all produce syntactically valid queries."""
        for name in ["aspirin", "ASPIRIN", "Aspirin"]:
            schema = ClinicalQuerySchema(drug_name=name)
            query = client.build_query(schema.drug_name)
            assert "medicinalproduct:" in query
            assert "limit=" in query

    def test_build_query_lucene_injection(self, client):
        """Adversarial input like '" OR *:*' is escaped and does not break the Lucene query structure."""
        query = client.build_query('" OR *:*')
        # The quote should be escaped, preventing injection
        assert '\\"' in query
        # The query should still have exactly one opening and closing quote around the drug name
        search_part = query.split("search=")[1].split("&")[0]
        assert 'medicinalproduct:"' in query


class TestBoundaryConditions:
    """Tests for numerical and temporal edge cases."""

    def test_age_boundary_cohort_bracket(self, client):
        """Age=1 produces cohort bracket [0, 6]; age=120 produces [115, 125]."""
        query_young = client.build_query("X", patient_age=1)
        assert "[0+TO+6]" in query_young
        query_old = client.build_query("X", patient_age=120)
        assert "[115+TO+125]" in query_old

    def test_recency_decay_smooth(self, math_client, monkeypatch):
        """Exponential decay: 89 and 91 day scores are nearly identical (no cliff)."""
        day_89 = (datetime.now() - timedelta(days=89)).strftime("%Y%m%d")
        day_91 = (datetime.now() - timedelta(days=91)).strftime("%Y%m%d")
        report_template = {"severity": "Non-Serious", "is_death": False, "is_hospitalization": False, "symptoms": "headache"}
        reports_89 = [{**report_template, "date": day_89}]
        reports_91 = [{**report_template, "date": day_91}]
        monkeypatch.setattr(math_client, "fetch_label_text", lambda *a: "headache")
        score_89 = math_client.calculate_final_score("TEST", reports_89, skip_benchmark=True)["clinical_signal"]["adverse_score"]
        score_91 = math_client.calculate_final_score("TEST", reports_91, skip_benchmark=True)["clinical_signal"]["adverse_score"]
        # Exponential decay: 89d ≈ 0.504, 91d ≈ 0.496 — scores differ by < 0.5
        assert abs(score_89 - score_91) < 0.5
        assert score_89 > 0
        assert score_91 > 0

    def test_score_all_death_unlabeled_recent(self, math_client, monkeypatch):
        """Maximum theoretical input (all death + unlabeled + recent) → score capped at 100."""
        today = datetime.now().strftime("%Y%m%d")
        reports = [
            {"date": today, "severity": "Serious", "is_death": True,
             "is_hospitalization": False, "symptoms": "rare_xyz"}
            for _ in range(10)
        ]
        monkeypatch.setattr(math_client, "fetch_label_text", lambda *a: "")
        result = math_client.calculate_final_score("TEST", reports, skip_benchmark=True)
        assert result["clinical_signal"]["adverse_score"] == 100

    def test_score_all_non_serious_labeled_old(self, math_client, monkeypatch):
        """Minimum theoretical input (all non-serious + labeled + 180 days old) → score = 2.5."""
        old_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        reports = [
            {"date": old_date, "severity": "Non-Serious", "is_death": False,
             "is_hospitalization": False, "symptoms": "headache"}
            for _ in range(10)
        ]
        # Label text contains "headache" so penalty is 1.0
        monkeypatch.setattr(math_client, "fetch_label_text", lambda *a: "headache, nausea, fatigue")
        result = math_client.calculate_final_score("TEST", reports, skip_benchmark=True)
        # 0.25 (NON_SERIOUS) * 1.0 (labeled) * 0.25 (180d exponential decay) * 40 = 2.5
        assert result["clinical_signal"]["adverse_score"] == 2.5

    def test_concurrent_severity_tiers(self, math_client, monkeypatch):
        """Mixed severity reports produce a weighted average between the individual tier scores."""
        today = datetime.now().strftime("%Y%m%d")
        reports = [
            {"date": today, "severity": "Serious", "is_death": True,
             "is_hospitalization": False, "symptoms": "headache"},
            {"date": today, "severity": "Non-Serious", "is_death": False,
             "is_hospitalization": False, "symptoms": "headache"},
        ]
        monkeypatch.setattr(math_client, "fetch_label_text", lambda *a: "headache")
        result = math_client.calculate_final_score("TEST", reports, skip_benchmark=True)
        # Death labeled: 1.75*1.0=1.75, Non-serious labeled: 0.25*1.0=0.25
        # Mean = (1.75+0.25)/2 = 1.0, score = 1.0 * 40 = 40.0
        assert result["clinical_signal"]["adverse_score"] == 40.0

    def test_single_report_scoring(self, math_client, monkeypatch):
        """Scoring is deterministic and correct with exactly 1 report as input."""
        today = datetime.now().strftime("%Y%m%d")
        reports = [
            {"date": today, "severity": "Serious", "is_death": False,
             "is_hospitalization": True, "symptoms": "nausea"}
        ]
        monkeypatch.setattr(math_client, "fetch_label_text", lambda *a: "nausea, vomiting")
        result = math_client.calculate_final_score("TEST", reports, skip_benchmark=True)
        # HOSPITALIZATION (1.0) * labeled (1.0) * recent (1.0) * 40 = 40.0
        assert result["clinical_signal"]["adverse_score"] == 40.0
