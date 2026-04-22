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


# ── Clinical Scoring Constants ─────────────���──────────────────────────────
# Severity tier weights for individual adverse event reports.
# Based on regulatory seriousness criteria (ICH E2D guidelines).
SEVERITY_WEIGHT_DEATH = 1.75           # Fatal outcome — highest signal weight
SEVERITY_WEIGHT_HOSPITALIZATION = 1.0  # Required or prolonged hospitalization
SEVERITY_WEIGHT_OTHER_SERIOUS = 0.75   # Serious but not hospitalized (disability, congenital anomaly, etc.)
SEVERITY_WEIGHT_NON_SERIOUS = 0.25     # Non-serious adverse event

# Label penalty multipliers — unlabeled events receive higher weight because
# they represent potentially novel safety signals not yet in the drug label.
LABEL_PENALTY_UNLABELED_SERIOUS = 2.0      # Serious + not in label → 2x weight
LABEL_PENALTY_UNLABELED_NON_SERIOUS = 1.5  # Non-serious + not in label → 1.5x weight
LABEL_PENALTY_LABELED = 1.0                # Already in label → no penalty

# Recency decay — exponential half-life model (replaces binary 90-day cliff).
# Reports lose half their weight every 90 days: weight = exp(-0.693 * days / 90).
RECENCY_HALF_LIFE_DAYS = 90
RECENCY_DECAY_CONSTANT = -0.693  # ln(0.5), mathematically derived from half-life

# Score normalization — final_score = min(100, mean_weighted_signal * SCORE_SCALAR).
# 40 was calibrated so that a drug with all-hospitalization labeled reports scores ~40/100.
SCORE_SCALAR = 40

# ── Confidence Curve Constants ────────────────────────────────���───────────
# Continuous log-linear curve: min(MAX, BASE + RANGE * log1p(N) / log1p(REF_N))
CONFIDENCE_MAX = 100.0
CONFIDENCE_BASE = 40.0                 # Minimum confidence for N=1
CONFIDENCE_RANGE = 60.0                # Maximum additional confidence from sample size
CONFIDENCE_REFERENCE_N = 80            # Sample size where curve approaches saturation
CONFIDENCE_DEFECT_PENALTY_WEIGHT = 50.0  # Weight applied to defect ratio penalty

# Confidence level thresholds
CONFIDENCE_THRESHOLD_HIGH = 85
CONFIDENCE_THRESHOLD_MEDIUM = 65
CONFIDENCE_THRESHOLD_LOW = 40

# ── Guardrail Thresholds ─────────────────────────────────────────────────
# Deterministic flags that control AI behavior downstream.
HUMAN_REVIEW_THRESHOLD = 70            # Score > this → requires_human_review = True
LOW_CONFIDENCE_REVIEW_THRESHOLD = 40   # Score > this AND confidence == 'Low' → human review
SPECIALIST_ROUTING_THRESHOLD = 60      # Score > this → route_to_specialist = True

# ── Status Classification Thresholds ─────────────────────────────────────
HIGH_SIGNAL_THRESHOLD = 70             # Score > this → "High Signal - Urgent Review"
MONITOR_THRESHOLD = 30                 # Score > this → "Monitor - Emerging Trend"

# ── Benchmark Comparison Multipliers ────��────────────────────────────────
ELEVATED_RISK_MULTIPLIER = 1.5         # score > benchmark * 1.5 → "Elevated vs Class Peers"
LOWER_RISK_MULTIPLIER = 0.7            # score < benchmark * 0.7 → "Lower than Class Peers"

# ── PRR (Proportional Reporting Ratio) Constants ─────────────────────────
PRR_MINIMUM_DRUG_CASES = 3             # Minimum 'a' value for statistically valid PRR
PRR_Z_SCORE_95 = 1.96                  # Z-score for 95% Wald confidence interval
PRR_SIGNAL_THRESHOLD = 1.0             # CI lower bound must exceed this for signal detection

# ── Trend Classification Thresholds ────────���─────────────────────────────
TREND_RISING_THRESHOLD = 10            # Score delta >= this → RISING
TREND_DECLINING_THRESHOLD = -10        # Score delta <= this → DECLINING

# ── openFDA API Configuration ────────��───────────────────────────────────
API_TIMEOUT_DEFAULT = 10               # Seconds — standard endpoint timeout
API_TIMEOUT_AGGREGATION = 15           # Seconds — count endpoint (server-side aggregation)
DEFAULT_DAYS_BACK = 365                # Default lookback window for event queries
DEFAULT_EVENT_LIMIT = 500              # Max results per event query (openFDA caps at 1000)
DEFAULT_COUNT_LIMIT = 1000             # Max results for count endpoint queries
AGE_COHORT_RANGE = 5                   # +/- years for patient age bracket in queries
MAX_PEERS = 3                          # Maximum peer drugs for benchmark comparison
MIN_PEER_NAME_LENGTH = 3              # Exclude abbreviations (<=3 chars) from peer list
LABEL_FALLBACK_LIMIT = 5              # Max label results for class fallback lookup

# ── Retry Configuration ───────────��─────────────────────────────────────��
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