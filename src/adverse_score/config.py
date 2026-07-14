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

# ── Phase 5 — Deterministic Ranking Engine ────────────────────────────────
# Signal ranking is a lexicographic tiered sort across four criteria (Seriousness &
# Outcome, Strength of Evidence, Reversibility, Public Health Impact) — never a
# weighted sum or single composite scalar (explicitly banned by
# docs/PSUR_CONSOLIDATION_SCOPE.md Section 3.1). Each criterion below defines an
# ordered tuple of tier-name strings; ranking.py maps a signal's tier string to its
# position in the relevant tuple (index 0 = highest priority) purely as an internal
# sort key — that integer is never persisted on a RankedSignal or exposed as a score.

RANKING_FORMULA_VERSION = "1.0"  # Guardrail 6 audit-trail tag; must appear verbatim
                                  # in ranking.py's RankingResult.formula_version.

# Seriousness & Outcome tiers — ordinal clinical severity ordering, carried forward
# from the old (Phase-0-removed) scoring.py's SEVERITY_WEIGHTS relative ordering
# (DEATH > HOSPITALIZATION > OTHER_SERIOUS > NON_SERIOUS), but as pure ordinal tiers
# only — no numeric weights are reintroduced.
SERIOUSNESS_TIER_DEATH = "DEATH"
SERIOUSNESS_TIER_HOSPITALIZATION = "HOSPITALIZATION"
SERIOUSNESS_TIER_OTHER_SERIOUS = "OTHER_SERIOUS"
SERIOUSNESS_TIER_NON_SERIOUS = "NON_SERIOUS"
SERIOUSNESS_TIER_ORDER = (
    SERIOUSNESS_TIER_DEATH,
    SERIOUSNESS_TIER_HOSPITALIZATION,
    SERIOUSNESS_TIER_OTHER_SERIOUS,
    SERIOUSNESS_TIER_NON_SERIOUS,
)

# Strength-of-Evidence tiers — PRR signal_detected (calculate_prr's own boolean)
# crossed with label status (calculate_prr's own label_status field). "Strong" =
# signal_detected True; "Weak" = signal_detected False. Within each signal_detected
# bucket, UNLABELED outranks LABELED (an unlabeled signal carries more evidentiary
# weight per the scope doc), and LABEL_STATUS_UNKNOWN is its own lowest-priority
# bucket overall (least actionable — label status genuinely unknown).
STRENGTH_TIER_STRONG_UNLABELED = "STRONG_UNLABELED"
STRENGTH_TIER_STRONG_LABELED = "STRONG_LABELED"
STRENGTH_TIER_WEAK_UNLABELED = "WEAK_UNLABELED"
STRENGTH_TIER_WEAK_LABELED = "WEAK_LABELED"
STRENGTH_TIER_UNKNOWN_LABEL_STATUS = "UNKNOWN_LABEL_STATUS"
STRENGTH_TIER_ORDER = (
    STRENGTH_TIER_STRONG_UNLABELED,
    STRENGTH_TIER_STRONG_LABELED,
    STRENGTH_TIER_WEAK_UNLABELED,
    STRENGTH_TIER_WEAK_LABELED,
    STRENGTH_TIER_UNKNOWN_LABEL_STATUS,
)

# Reversibility tiers — derived from FAERS patient.reaction.reactionoutcome codes.
# Per openFDA's documented code set:
#   1 = Recovered/resolved
#   2 = Recovering/resolving
#   3 = Not recovered/not resolved
#   4 = Recovered/resolved with sequelae
#   5 = Fatal
#   6 = Unknown
# REVERSIBILITY is a documented heuristic (scope doc Section 7: FAERS structured
# fields cannot always directly establish true clinical reversibility) — codes are
# bucketed into four clinical tiers rather than used as six raw values.
REACTION_OUTCOME_RECOVERED = 1
REACTION_OUTCOME_RECOVERING = 2
REACTION_OUTCOME_NOT_RECOVERED = 3
REACTION_OUTCOME_RECOVERED_WITH_SEQUELAE = 4
REACTION_OUTCOME_FATAL = 5
REACTION_OUTCOME_UNKNOWN_CODE = 6

REVERSIBILITY_TIER_FATAL = "FATAL"
REVERSIBILITY_TIER_POOR = "POOR"
REVERSIBILITY_TIER_REVERSIBLE = "REVERSIBLE"
REVERSIBILITY_TIER_UNKNOWN = "UNKNOWN"
REVERSIBILITY_TIER_ORDER = (
    REVERSIBILITY_TIER_FATAL,
    REVERSIBILITY_TIER_POOR,
    REVERSIBILITY_TIER_REVERSIBLE,
    REVERSIBILITY_TIER_UNKNOWN,
)

# Public Health Impact tiers — report volume is used as a directional proxy for
# population exposure, per scope doc Section 7's documented simplification (FAERS
# report counts are not true epidemiological exposure data). Thresholds are round
# numbers chosen for a prototype-appropriate three-bucket split, not derived from
# an epidemiological model.
PUBLIC_HEALTH_HIGH_VOLUME_THRESHOLD = 100      # >= this many drug_cases -> HIGH
PUBLIC_HEALTH_MODERATE_VOLUME_THRESHOLD = 20   # >= this many (but < HIGH) -> MODERATE
                                                # below MODERATE threshold -> LOW
PUBLIC_HEALTH_TIER_HIGH = "HIGH"
PUBLIC_HEALTH_TIER_MODERATE = "MODERATE"
PUBLIC_HEALTH_TIER_LOW = "LOW"
PUBLIC_HEALTH_TIER_ORDER = (
    PUBLIC_HEALTH_TIER_HIGH,
    PUBLIC_HEALTH_TIER_MODERATE,
    PUBLIC_HEALTH_TIER_LOW,
)

# ── Phase 8 — Agent Orchestration & Guardrails ────────────────────────────
OPENAI_CHAT_MODEL = "gpt-4o"           # per scope doc Section 3.3, no engine change
AGENT_TEMPERATURE = 0.1                # low — favors consistency over creativity in a PV context
AGENT_MAX_ITERATIONS = 5               # caps the tool-calling loop; prevents runaway agent behavior
TOP_N_NARRATED_SIGNALS = 20            # scope doc's own example cap for conversational narration
CONVERSATION_ROLE_USER = "user"
CONVERSATION_ROLE_ASSISTANT = "assistant"
CONVERSATION_ROLE_SYSTEM = "system"

# ── Phase 9 — Document Generation (PBRER/PSUR .docx export) ──────────────
DOCUMENT_NARRATION_TEMPERATURE = 0.0   # lower than AGENT_TEMPERATURE — prioritizes faithfulness over natural variation for a document artifact
PBRER_PLACEHOLDER_TEXT = "[Section not populated by AdverseScore — to be completed by Regulatory Affairs]"
PBRER_DRAFT_MARKING_TEXT = "DRAFT — NOT FOR REGULATORY SUBMISSION — Pending Qualified PV/Clinical Review"
PBRER_OMITTED_SECTIONS_NOTE = "Sections 2-5, 7-14, 16.4-19, and 20 (Appendices) of the ICH E2C(R2) PBRER format are not populated by AdverseScore in this draft."

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
