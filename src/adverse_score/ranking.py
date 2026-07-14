from dataclasses import dataclass

from .config import (
    RANKING_FORMULA_VERSION,
    SERIOUSNESS_TIER_DEATH, SERIOUSNESS_TIER_HOSPITALIZATION,
    SERIOUSNESS_TIER_OTHER_SERIOUS, SERIOUSNESS_TIER_NON_SERIOUS,
    SERIOUSNESS_TIER_ORDER,
    STRENGTH_TIER_STRONG_UNLABELED, STRENGTH_TIER_STRONG_LABELED,
    STRENGTH_TIER_WEAK_UNLABELED, STRENGTH_TIER_WEAK_LABELED,
    STRENGTH_TIER_UNKNOWN_LABEL_STATUS, STRENGTH_TIER_ORDER,
    REACTION_OUTCOME_RECOVERED, REACTION_OUTCOME_RECOVERING,
    REACTION_OUTCOME_NOT_RECOVERED, REACTION_OUTCOME_RECOVERED_WITH_SEQUELAE,
    REACTION_OUTCOME_FATAL, REACTION_OUTCOME_UNKNOWN_CODE,
    REVERSIBILITY_TIER_FATAL, REVERSIBILITY_TIER_POOR,
    REVERSIBILITY_TIER_REVERSIBLE, REVERSIBILITY_TIER_UNKNOWN,
    REVERSIBILITY_TIER_ORDER,
    PUBLIC_HEALTH_HIGH_VOLUME_THRESHOLD, PUBLIC_HEALTH_MODERATE_VOLUME_THRESHOLD,
    PUBLIC_HEALTH_TIER_HIGH, PUBLIC_HEALTH_TIER_MODERATE, PUBLIC_HEALTH_TIER_LOW,
    PUBLIC_HEALTH_TIER_ORDER,
)
from .prr import calculate_prr
from .label_classifier import classify_label_statuses
from .logger import get_logger, log_event

logger = get_logger("ranking")


# ── Output Data Model ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RankedSignal:
    symptom: str               # normalized (uppercased) MedDRA PT term
    rank: int                  # 1-indexed position after lexicographic tiered sort
    seriousness_tier: str      # DEATH | HOSPITALIZATION | OTHER_SERIOUS | NON_SERIOUS
    strength_of_evidence_tier: str  # STRONG_UNLABELED | STRONG_LABELED | WEAK_UNLABELED |
                                    # WEAK_LABELED | UNKNOWN_LABEL_STATUS
    reversibility_tier: str    # FATAL | POOR | REVERSIBLE | UNKNOWN
    public_health_tier: str    # HIGH | MODERATE | LOW
    prr_metrics: dict          # verbatim calculate_prr() output for this symptom
    report_count: int          # == prr_metrics["drug_cases"]


@dataclass(frozen=True)
class RankingResult:
    ranked_signals: list       # list[RankedSignal], sorted by tiered priority, rank-assigned
    label_summary: object      # LabelClassificationResult — aggregate completeness statistic
    formula_version: str       # RANKING_FORMULA_VERSION, Guardrail 6 audit-trail tag
    total_signals: int         # len(ranked_signals)


# ── Per-Criterion Tier Derivation (pure functions, no HTTP) ──────────────

def _seriousness_tier_for_symptom(reports: list, symptom: str) -> str:
    """Worst-case seriousness tier across every report mentioning `symptom` (already
    normalized/uppercased by the caller). 'Worst' = highest-priority position in
    SERIOUSNESS_TIER_ORDER (index 0 = DEATH, the most severe)."""
    worst_rank = len(SERIOUSNESS_TIER_ORDER) - 1  # start at least-severe
    for report in reports:
        report_symptoms = {s.strip().upper() for s in (report.get("symptom_list") or []) if s and s.strip()}
        if symptom not in report_symptoms:
            continue

        if report.get("is_death"):
            tier = SERIOUSNESS_TIER_DEATH
        elif report.get("is_hospitalization"):
            tier = SERIOUSNESS_TIER_HOSPITALIZATION
        elif report.get("severity") == "Serious":
            tier = SERIOUSNESS_TIER_OTHER_SERIOUS
        else:
            tier = SERIOUSNESS_TIER_NON_SERIOUS

        rank = SERIOUSNESS_TIER_ORDER.index(tier)
        if rank < worst_rank:
            worst_rank = rank

    return SERIOUSNESS_TIER_ORDER[worst_rank]


def _strength_of_evidence_tier(signal_detected: bool, label_status: str) -> str:
    """Maps calculate_prr's signal_detected boolean + label_status string to a single
    Strength-of-Evidence tier. LABEL_STATUS_UNKNOWN is its own lowest-priority bucket
    regardless of signal_detected — label status is genuinely unknown, so we cannot
    honor the "unlabeled outranks labeled" modifier at all in that case."""
    if label_status == "LABEL_STATUS_UNKNOWN":
        return STRENGTH_TIER_UNKNOWN_LABEL_STATUS
    if signal_detected:
        return STRENGTH_TIER_STRONG_UNLABELED if label_status == "UNLABELED" else STRENGTH_TIER_STRONG_LABELED
    return STRENGTH_TIER_WEAK_UNLABELED if label_status == "UNLABELED" else STRENGTH_TIER_WEAK_LABELED


def _reversibility_tier_for_symptom(reports: list, symptom: str) -> str:
    """Worst-case reversibility tier across every reactionoutcome code found on any
    `reactions` entry (in any report mentioning `symptom`) whose term normalizes to
    `symptom`. If NO matching reactions entry anywhere has a non-None outcome_code,
    the tier is UNKNOWN — reversibility is never assumed/defaulted to REVERSIBLE
    absent real data (documented heuristic per scope doc Section 7)."""
    codes_seen = []
    for report in reports:
        report_symptoms = {s.strip().upper() for s in (report.get("symptom_list") or []) if s and s.strip()}
        if symptom not in report_symptoms:
            continue
        for reaction in (report.get("reactions") or []):
            term = reaction.get("term")
            if term and term.strip().upper() == symptom:
                code = reaction.get("outcome_code")
                if code is not None:
                    codes_seen.append(code)

    if not codes_seen:
        return REVERSIBILITY_TIER_UNKNOWN

    def _code_tier(code: int) -> str:
        if code == REACTION_OUTCOME_FATAL:
            return REVERSIBILITY_TIER_FATAL
        if code in (REACTION_OUTCOME_NOT_RECOVERED, REACTION_OUTCOME_RECOVERED_WITH_SEQUELAE):
            return REVERSIBILITY_TIER_POOR
        if code in (REACTION_OUTCOME_RECOVERED, REACTION_OUTCOME_RECOVERING):
            return REVERSIBILITY_TIER_REVERSIBLE
        return REVERSIBILITY_TIER_UNKNOWN  # REACTION_OUTCOME_UNKNOWN_CODE (6)

    worst_rank = len(REVERSIBILITY_TIER_ORDER) - 1
    for code in codes_seen:
        rank = REVERSIBILITY_TIER_ORDER.index(_code_tier(code))
        if rank < worst_rank:
            worst_rank = rank

    return REVERSIBILITY_TIER_ORDER[worst_rank]


def _public_health_tier(drug_cases: int) -> str:
    """Report volume (drug_cases from this signal's own PRR metrics) as a directional
    exposure proxy, per scope doc Section 7's documented simplification."""
    if drug_cases >= PUBLIC_HEALTH_HIGH_VOLUME_THRESHOLD:
        return PUBLIC_HEALTH_TIER_HIGH
    if drug_cases >= PUBLIC_HEALTH_MODERATE_VOLUME_THRESHOLD:
        return PUBLIC_HEALTH_TIER_MODERATE
    return PUBLIC_HEALTH_TIER_LOW


# ── Entry Point ────────────────────────────────────────────────────────────

def rank_signals(reports: list, class_counts: dict, label_text: str) -> RankingResult:
    """Deterministic, lexicographic-tiered ranking of adverse-event signals (one
    signal = one unique MedDRA PT symptom for the drug within the PSUR period).

    Combines four criteria — Seriousness & Outcome, Strength of Evidence,
    Reversibility, Public Health Impact — via tuple-of-tier-strings sort, NEVER a
    weighted sum or single composite scalar (explicitly banned by
    docs/PSUR_CONSOLIDATION_SCOPE.md Section 3.1). Makes no HTTP calls and does not
    import FDAClient — `reports` and `class_counts` must be pre-fetched by the caller.

    Args:
        reports: deduplicated flattened report list (deduplication.py's DedupResult.reports
            shape — has symptom_list, reactions, severity, is_death, is_hospitalization,
            drug_names, date, etc.)
        class_counts: dict, uppercase symptom -> peer/background mention count (same
            shape prr.py's calculate_prr already expects).
        label_text: passed through to calculate_prr per signal.
    """
    # 1. Unique uppercase symptom set across all reports.
    unique_symptoms = sorted({
        s.strip().upper()
        for report in reports
        for s in (report.get("symptom_list") or [])
        if s and s.strip()
    })

    if not unique_symptoms:
        empty_label_summary = classify_label_statuses(label_text, [])
        return RankingResult(
            ranked_signals=[],
            label_summary=empty_label_summary,
            formula_version=RANKING_FORMULA_VERSION,
            total_signals=0,
        )

    # 2. Mention-count drug_counts (matches class_counts' mention-based convention —
    # a single report can list multiple symptoms, so this counts symptom mentions,
    # not distinct reports). In the same pass, build a symptom -> matching-reports
    # index so per-symptom tier derivation below is O(total mentions) instead of
    # O(unique_symptoms x reports) full rescans — at real PSUR volume (tens of
    # thousands of reports, thousands of unique PTs) the rescan approach is
    # prohibitively slow.
    drug_counts: dict = {}
    symptom_to_reports: dict = {}
    for report in reports:
        normalized_mentions = [
            s.strip().upper() for s in (report.get("symptom_list") or []) if s and s.strip()
        ]
        for key in normalized_mentions:
            drug_counts[key] = drug_counts.get(key, 0) + 1
        for key in set(normalized_mentions):
            symptom_to_reports.setdefault(key, []).append(report)

    # 3. Aggregate label summary/completeness statistic (distinct from each signal's
    # own per-symptom label status derived below via calculate_prr).
    label_summary = classify_label_statuses(label_text, unique_symptoms)

    # 4. Per-symptom tier derivation, in sorted order for determinism.
    signals = []
    for symptom in unique_symptoms:
        prr_metrics = calculate_prr(drug_counts, class_counts, symptom, label_text)

        strength_tier = _strength_of_evidence_tier(
            prr_metrics["signal_detected"], prr_metrics["label_status"])
        matching_reports = symptom_to_reports.get(symptom, [])
        seriousness_tier = _seriousness_tier_for_symptom(matching_reports, symptom)
        reversibility_tier = _reversibility_tier_for_symptom(matching_reports, symptom)
        public_health_tier = _public_health_tier(prr_metrics["drug_cases"])

        signals.append(RankedSignal(
            symptom=symptom,
            rank=0,  # assigned after sort
            seriousness_tier=seriousness_tier,
            strength_of_evidence_tier=strength_tier,
            reversibility_tier=reversibility_tier,
            public_health_tier=public_health_tier,
            prr_metrics=prr_metrics,
            report_count=prr_metrics["drug_cases"],
        ))

    # 5. Lexicographic tiered sort: seriousness, strength-of-evidence, reversibility,
    # public-health, in that fixed priority order. Tier-order ints are internal sort
    # keys only — never stored on RankedSignal/RankingResult.
    def _sort_key(signal: RankedSignal) -> tuple:
        return (
            SERIOUSNESS_TIER_ORDER.index(signal.seriousness_tier),
            STRENGTH_TIER_ORDER.index(signal.strength_of_evidence_tier),
            REVERSIBILITY_TIER_ORDER.index(signal.reversibility_tier),
            PUBLIC_HEALTH_TIER_ORDER.index(signal.public_health_tier),
        )

    signals.sort(key=_sort_key)

    # 6. Assign 1-indexed rank post-sort.
    ranked_signals = []
    for i, signal in enumerate(signals, start=1):
        ranked_signals.append(RankedSignal(
            symptom=signal.symptom,
            rank=i,
            seriousness_tier=signal.seriousness_tier,
            strength_of_evidence_tier=signal.strength_of_evidence_tier,
            reversibility_tier=signal.reversibility_tier,
            public_health_tier=signal.public_health_tier,
            prr_metrics=signal.prr_metrics,
            report_count=signal.report_count,
        ))

    log_event(logger, "ranking_complete", total_signals=len(ranked_signals),
              formula_version=RANKING_FORMULA_VERSION)

    return RankingResult(
        ranked_signals=ranked_signals,
        label_summary=label_summary,
        formula_version=RANKING_FORMULA_VERSION,
        total_signals=len(ranked_signals),
    )
