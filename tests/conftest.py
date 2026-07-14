import os
import time
import pytest
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env so API keys are available for skip-guard evaluation
load_dotenv()


# ── E2E Skip Guards ────────────────────────────────────────────────────────
# Applied to E2E tests so they skip cleanly when API keys are absent.

SKIP_NO_FDA = pytest.mark.skipif(
    not os.environ.get("OPENFDA_API_KEY"),
    reason="OPENFDA_API_KEY not set — skipping live FDA API test"
)
SKIP_NO_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping LLM agent test"
)


# ── E2E Session-Scoped Client ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def e2e_client():
    """Session-scoped AdverseScoreClient for E2E tests (reuses HTTP session)."""
    from adverse_score.client import AdverseScoreClient
    return AdverseScoreClient()


# ── E2E Rate Limit Guard ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def rate_limit_guard(request):
    """Inserts a 0.3s delay after each E2E test to respect openFDA rate limits."""
    yield
    if request.node.get_closest_marker("e2e"):
        time.sleep(0.3)


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
                        {"reactionmeddrapt": "CARDIAC ARREST", "reactionoutcome": "5"},
                        {"reactionmeddrapt": "DEATH", "reactionoutcome": "5"},
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


# ── FIXTURE: Persistence Store (temp SQLite) ─────────────────────────────

@pytest.fixture
def temp_store(tmp_path):
    """Provides a ConsolidationStore backed by a temporary SQLite DB."""
    from adverse_score.persistence import ConsolidationStore
    db_path = tmp_path / "test.db"
    store = ConsolidationStore(db_path=db_path)
    yield store
    store.close()
