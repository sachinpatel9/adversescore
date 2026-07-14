import os
from dotenv import load_dotenv

def initialize_config() -> str:
    """
    Loads environment variables and validates the presence of all required API keys.
    Returns the openFDA API key for the AdverseScoreClient.
    """
    load_dotenv()

    fda_key = os.getenv('OPENFDA_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')

    # Fail Fast Validations
    if not fda_key:
        raise EnvironmentError("OPENFDA_API_KEY is not set in environment variables. Please set it in your .env file.")

    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. The LangChain Agent requires this to execute.")

    return fda_key


# ── PRR (Proportional Reporting Ratio) Constants ─────────────────────────
PRR_MINIMUM_DRUG_CASES = 3             # Minimum 'a' value for statistically valid PRR
PRR_Z_SCORE_95 = 1.96                  # Z-score for 95% Wald confidence interval
PRR_SIGNAL_THRESHOLD = 1.0             # CI lower bound must exceed this for signal detection

# ── openFDA API Configuration ─────────────────────────────────────────────
API_TIMEOUT_DEFAULT = 10               # Seconds — standard endpoint timeout
API_TIMEOUT_AGGREGATION = 15           # Seconds — count endpoint (server-side aggregation)
DEFAULT_DAYS_BACK = 365                # Default lookback window for event queries
DEFAULT_EVENT_LIMIT = 500              # Max results per event query (openFDA caps at 1000)
DEFAULT_COUNT_LIMIT = 1000             # Max results for count endpoint queries
AGE_COHORT_RANGE = 5                   # +/- years for patient age bracket in queries
MAX_PEERS = 3                          # Maximum peer drugs for benchmark comparison
MIN_PEER_NAME_LENGTH = 3              # Exclude abbreviations (<=3 chars) from peer list
LABEL_FALLBACK_LIMIT = 5              # Max label results for class fallback lookup

# ── Drug Identity Resolution Constants (Phase 1) ─────────────────────────
# Endpoint paths for drug identity/approval-date resolution — kept as named
# constants so a future openFDA API version bump is a one-line change.
DRUGSFDA_ENDPOINT = "https://api.fda.gov/drug/drugsfda.json"
NDC_ENDPOINT = "https://api.fda.gov/drug/ndc.json"

# Result caps for identity-resolution queries. Small limits are intentional:
# we need enough records to union brand/generic/substance name variants and
# find an application_number, not an exhaustive product catalog.
DRUG_IDENTITY_LABEL_LIMIT = 5          # max label.json records to inspect per lookup
DRUG_IDENTITY_NDC_LIMIT = 5            # max ndc.json records to inspect per fallback lookup
DRUG_IDENTITY_DRUGSFDA_LIMIT = 10      # max drugsfda.json application records per date lookup

# submission_status value considered "approved" for market authorization date
# purposes. Other values (e.g. "TA" tentative approval, "WD" withdrawn) are
# excluded — a withdrawn-then-reapproved product's history is not modeled.
DRUGSFDA_APPROVED_STATUS = "AP"

# ── PSUR Chunked Retrieval Constants (Phase 2) ────────────────────────────
PSUR_PAGE_SIZE = 1000          # openFDA's hard per-request cap on `limit`. Distinct from
                                # DEFAULT_EVENT_LIMIT=500 (single-page sampling) — PSUR retrieval
                                # must exhaust each chunk, so always request the API's real max.
PSUR_SKIP_CEILING = 25000      # openFDA enforces skip+limit <= 26000. Capping skip at 25000
                                # leaves room for one final PSUR_PAGE_SIZE=1000 page (25000+1000=
                                # 26000, exactly at the boundary) before flagging truncation.
PSUR_CHUNK_MONTHS = 3          # Sub-period chunk width (quarterly), counted forward from the
                                # PSUR period's anchor date — not calendar-quarter-aligned.
PSUR_PERIOD_MONTHS = {"6mo": 6, "1yr": 12, "2yr": 24, "3yr": 36}
PSUR_PERIOD_FALLBACK_DAYS = {"6mo": 182, "1yr": 365, "2yr": 730, "3yr": 1095}  # rolling-lookback
                                # day counts used only on the fallback (no-anchor) path.

# ── Retry Configuration ────────────────────────────────────────────────────
RETRY_TOTAL = 3                        # urllib3 transport-level retry count
RETRY_BACKOFF_FACTOR = 1               # urllib3 exponential backoff multiplier
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]  # HTTP codes that trigger transport retry
TENACITY_MAX_ATTEMPTS = 3              # Application-level retry attempts
TENACITY_WAIT_MULTIPLIER = 1           # Tenacity exponential backoff multiplier
TENACITY_WAIT_MIN = 2                  # Minimum wait between retries (seconds)
TENACITY_WAIT_MAX = 10                 # Maximum wait between retries (seconds)


# Execution block for testing the config directly
if __name__ == '__main__':
    key = initialize_config()
    import logging, json, sys
    logging.basicConfig(stream=sys.stderr, format='%(message)s')
    logging.info(json.dumps({"event": "config_initialized"}))
