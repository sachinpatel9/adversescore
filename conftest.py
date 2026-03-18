import pytest
from datetime import datetime, timedelta


# ── FIXTURE: AdverseScoreClient Instance ─────────────────────────────────────
# Requires a valid .env with OPENFDA_API_KEY and OPENAI_API_KEY.
# Used by integration tests and math tests that call client methods directly.

@pytest.fixture
def client():
    """Provides a live AdverseScoreClient instance with a real HTTP session."""
    from adverse_score.client import AdverseScoreClient
    return AdverseScoreClient()


# ── FIXTURE: Drug Name Categories ────────────────────────────────────────────

@pytest.fixture
def sample_drug_names():
    """Dict of drug names organized by test category."""
    return {
        "common": "KEYTRUDA",
        "obscure": "DANTROLENE",
        "misspelled": "ASPIRN",
        "special_slash": "INSULIN/DEXTROSE",
        "special_hyphen": "L-DOPA",
        "special_quotes": 'DRUG"NAME',
        "special_backslash": "DRUG\\NAME",
    }


# ── FIXTURE: Mock openFDA event.json Response ────────────────────────────────
# Realistic response with 4 reports covering all severity tiers.

@pytest.fixture
def mock_fda_response():
    """A realistic openFDA event.json response with 4 reports spanning all severity tiers."""
    today = datetime.now().strftime("%Y%m%d")
    old_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")

    return {
        "meta": {"results": {"total": 4, "skip": 0, "limit": 500}},
        "results": [
            {
                "safetyreportid": "RPT-001",
                "receivedate": today,
                "seriousness": "1",
                "seriousnessdeath": "1",
                "seriousnesshospitalization": None,
                "patient": {
                    "reaction": [
                        {"reactionmeddrapt": "CARDIAC ARREST"},
                        {"reactionmeddrapt": "DEATH"},
                    ]
                },
                "companynumb": "PHARMA-001",
            },
            {
                "safetyreportid": "RPT-002",
                "receivedate": today,
                "seriousness": "1",
                "seriousnessdeath": None,
                "seriousnesshospitalization": "1",
                "patient": {
                    "reaction": [
                        {"reactionmeddrapt": "HEPATOTOXICITY"},
                    ]
                },
                "companynumb": "PHARMA-002",
            },
            {
                "safetyreportid": "RPT-003",
                "receivedate": old_date,
                "seriousness": "1",
                "seriousnessdeath": None,
                "seriousnesshospitalization": None,
                "patient": {
                    "reaction": [
                        {"reactionmeddrapt": "NAUSEA"},
                        {"reactionmeddrapt": "FATIGUE"},
                    ]
                },
                "companynumb": "PHARMA-003",
            },
            {
                "safetyreportid": "RPT-004",
                "receivedate": old_date,
                "seriousness": None,
                "seriousnessdeath": None,
                "seriousnesshospitalization": None,
                "patient": {
                    "reaction": [
                        {"reactionmeddrapt": "HEADACHE"},
                    ]
                },
                "companynumb": "PHARMA-004",
            },
        ],
    }


# ── FIXTURE: Pre-Flattened Clean Reports ─────────────────────────────────────
# Output of _flatten_results with controlled values for deterministic math tests.

@pytest.fixture
def sample_clean_reports():
    """Pre-flattened reports with known severity/date/symptom distributions."""
    today = datetime.now().strftime("%Y%m%d")
    old_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")

    return [
        {
            "report_id": "RPT-001",
            "date": today,
            "severity": "Serious",
            "is_death": True,
            "is_hospitalization": False,
            "symptoms": "CARDIAC ARREST, DEATH",
            "company": "PHARMA-001",
        },
        {
            "report_id": "RPT-002",
            "date": today,
            "severity": "Serious",
            "is_death": False,
            "is_hospitalization": True,
            "symptoms": "HEPATOTOXICITY",
            "company": "PHARMA-002",
        },
        {
            "report_id": "RPT-003",
            "date": old_date,
            "severity": "Serious",
            "is_death": False,
            "is_hospitalization": False,
            "symptoms": "NAUSEA, FATIGUE",
            "company": "PHARMA-003",
        },
        {
            "report_id": "RPT-004",
            "date": old_date,
            "severity": "Non-Serious",
            "is_death": False,
            "is_hospitalization": False,
            "symptoms": "HEADACHE",
            "company": "PHARMA-004",
        },
    ]


# ── FIXTURE: Empty FDA Response ──────────────────────────────────────────────

@pytest.fixture
def mock_fda_empty_response():
    """Simulates openFDA returning None (404 / zero results)."""
    return None


# ── FIXTURE: Mock Label Response ─────────────────────────────────────────────

@pytest.fixture
def mock_label_response():
    """A realistic drug/label.json response with adverse_reactions text."""
    return {
        "results": [
            {
                "adverse_reactions": [
                    "The following adverse reactions have been reported: nausea, fatigue, headache, hepatotoxicity, rash."
                ]
            }
        ]
    }


# ── FIXTURE: Mock Symptom Counts ─────────────────────────────────────────────

@pytest.fixture
def mock_symptom_counts():
    """Known symptom count dicts for deterministic PRR calculation."""
    return {
        "drug_counts": {
            "NAUSEA": 50,
            "FATIGUE": 30,
            "HEADACHE": 20,
            "HEPATOTOXICITY": 10,
        },
        "class_counts": {
            "NAUSEA": 500,
            "FATIGUE": 400,
            "HEADACHE": 300,
            "HEPATOTOXICITY": 20,
        },
    }


# ── FIXTURE: Complete Agent Payload ──────────────────────────────────────────

@pytest.fixture
def sample_agent_payload():
    """A complete calculate_final_score output with all required fields."""
    return {
        "metadata": {
            "tool_name": "AdverseScore",
            "version": "1.0",
            "timestamp": datetime.now().isoformat() + "Z",
            "clinical_disclaimer": "This score is for informational purposes only and does not constitute medical advice. The responsibility ultimately remains with the clinician.",
        },
        "clinical_signal": {
            "drug_target": "KEYTRUDA",
            "adverse_score": 45.0,
            "status": "Monitor - Emerging Trend for KEYTRUDA",
            "relative_risk": "Average",
            "class_benchmark_avg": 42.0,
            "label_status": "UNLABELED",
        },
        "data_integrity": {
            "report_count": 150,
            "confidence_level": "High",
            "defect_ratio": 0.05,
        },
        "pharmacovigilance_metrics": None,
        "agent_directives": {
            "diagnosis_lock": True,
            "requires_human_review": False,
            "route_to_specialist": False,
            "system_directive": "Halt autonomous clinical advice if requires_human_review is True.",
        },
    }


# ── FIXTURE: Agent Payload With PRR Metrics ──────────────────────────────────

@pytest.fixture
def sample_agent_payload_with_prr(sample_agent_payload):
    """A complete payload with pharmacovigilance_metrics populated including label_status."""
    sample_agent_payload["pharmacovigilance_metrics"] = {
        "prr": 2.5,
        "ci_lower": 1.8,
        "signal_detected": True,
        "target_symptom": "HEPATOTOXICITY",
        "drug_cases": 10,
        "class_cases": 20,
        "label_status": "LABELED",
    }
    return sample_agent_payload


# ── FIXTURE: Error Payload ───────────────────────────────────────────────────

@pytest.fixture
def sample_error_payload():
    """The error payload structure from agent_tools.py exception handler."""
    return {
        "metadata": {
            "tool_name": "AdverseScore",
            "status": "System Error",
            "clinical_disclaimer": "This tool is for informational purposes only and does not constitute medical advice.",
        },
        "agent_directives": {
            "diagnosis_lock": True,
            "requires_human_review": False,
            "route_to_specialist": False,
            "system_directive": "Inform the user that the AdverseScore tool encountered a system error and could not complete the analysis. Internal Error: test error",
        },
    }


# ── FIXTURE: Signal Narrative ────────────────────────────────────────────────

@pytest.fixture
def sample_narrative_keywords():
    """The set of keywords that trigger narrative generation."""
    return {"write", "narrative", "document", "report", "summarise", "summarize", "memo", "draft"}


@pytest.fixture
def sample_narrative_output():
    """A mock LLM response containing a Signal Evaluation Narrative with ICH E2E sections."""
    return (
        "## Metformin — AdverseScore: 45/100\n\n"
        "Some standard analysis content here.\n\n"
        "<!-- SIGNAL_NARRATIVE_START -->\n"
        "DRAFT — For Human Review Only\n\n"
        "### 1. Signal Description\n"
        "Metformin hydrochloride was identified with an AdverseScore of 45.\n\n"
        "### 2. Data Source and Method\n"
        "Data sourced from FDA Adverse Event Reporting System (FAERS) via openFDA API.\n\n"
        "### 3. Statistical Analysis Summary\n"
        "PRR of 2.5 (95% CI lower bound: 1.8) indicates disproportionate reporting.\n\n"
        "### 4. Clinical Assessment\n"
        "The signal pattern suggests elevated reporting relative to class peers. "
        "Note: this interpretation requires clinical validation.\n\n"
        "### 5. Data Limitations\n"
        "Report count of 150 provides medium confidence. FAERS is subject to reporting bias.\n\n"
        "### 6. Recommendation\n"
        "Recommend clinician review of the emerging signal pattern.\n\n"
        "This tool is for informational purposes only and does not constitute medical advice.\n"
        "<!-- SIGNAL_NARRATIVE_END -->\n"
    )


# ── FIXTURE: Temporal Trend Analysis ─────────────────────────────────────────

@pytest.fixture
def sample_time_series_rising():
    """A time_series array exhibiting a RISING trend (delta >= 10)."""
    return [
        {"quarter": "2025-Q2", "adverse_score": 30, "prr": 1.5, "report_count": 100},
        {"quarter": "2025-Q3", "adverse_score": 33, "prr": 1.7, "report_count": 110},
        {"quarter": "2025-Q4", "adverse_score": 38, "prr": 2.0, "report_count": 120},
        {"quarter": "2026-Q1", "adverse_score": 48, "prr": 2.5, "report_count": 85},
    ]


@pytest.fixture
def sample_time_series_stable():
    """A time_series array exhibiting a STABLE trend (delta < 10)."""
    return [
        {"quarter": "2025-Q2", "adverse_score": 40, "prr": 1.8, "report_count": 100},
        {"quarter": "2025-Q3", "adverse_score": 42, "prr": 1.9, "report_count": 105},
        {"quarter": "2025-Q4", "adverse_score": 41, "prr": 1.8, "report_count": 110},
        {"quarter": "2026-Q1", "adverse_score": 43, "prr": 1.9, "report_count": 95},
    ]


@pytest.fixture
def sample_time_series_declining():
    """A time_series array exhibiting a DECLINING trend (delta <= -10)."""
    return [
        {"quarter": "2025-Q2", "adverse_score": 55, "prr": 2.8, "report_count": 130},
        {"quarter": "2025-Q3", "adverse_score": 50, "prr": 2.4, "report_count": 125},
        {"quarter": "2025-Q4", "adverse_score": 45, "prr": 2.0, "report_count": 115},
        {"quarter": "2026-Q1", "adverse_score": 40, "prr": 1.6, "report_count": 100},
    ]


@pytest.fixture
def sample_time_series_insufficient():
    """A time_series array with insufficient data (only 1 valid quarter)."""
    return [
        {"quarter": "2025-Q2", "adverse_score": 0, "prr": None, "report_count": 0},
        {"quarter": "2025-Q3", "adverse_score": 0, "prr": None, "report_count": 0},
        {"quarter": "2025-Q4", "adverse_score": 0, "prr": None, "report_count": 0},
        {"quarter": "2026-Q1", "adverse_score": 42, "prr": 1.9, "report_count": 85},
    ]


@pytest.fixture
def sample_agent_payload_with_temporal(sample_agent_payload, sample_time_series_rising):
    """A complete payload with temporal_analysis populated."""
    sample_agent_payload["temporal_analysis"] = {
        "time_series": sample_time_series_rising,
        "trend_classification": "RISING",
    }
    return sample_agent_payload


# ── FIXTURE: Persistence Store (temp SQLite) ─────────────────────────────

@pytest.fixture
def temp_store(tmp_path):
    """Provides an AnalysisStore backed by a temporary SQLite DB."""
    from adverse_score.persistence import AnalysisStore
    db_path = tmp_path / "test.db"
    store = AnalysisStore(db_path=db_path)
    yield store
    store.close()
