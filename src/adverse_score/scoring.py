import math
from datetime import datetime, timedelta
from typing import Optional
from .config import (
    SEVERITY_WEIGHT_DEATH, SEVERITY_WEIGHT_HOSPITALIZATION,
    SEVERITY_WEIGHT_OTHER_SERIOUS, SEVERITY_WEIGHT_NON_SERIOUS,
    RECENCY_HALF_LIFE_DAYS, RECENCY_DECAY_CONSTANT, SCORE_SCALAR,
    CONFIDENCE_MAX, CONFIDENCE_BASE, CONFIDENCE_RANGE, CONFIDENCE_REFERENCE_N,
    CONFIDENCE_DEFECT_PENALTY_WEIGHT,
    CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM, CONFIDENCE_THRESHOLD_LOW,
    HUMAN_REVIEW_THRESHOLD, LOW_CONFIDENCE_REVIEW_THRESHOLD,
    SPECIALIST_ROUTING_THRESHOLD,
    HIGH_SIGNAL_THRESHOLD, MONITOR_THRESHOLD,
    ELEVATED_RISK_MULTIPLIER, LOWER_RISK_MULTIPLIER,
)
from .label_classifier import calculate_label_penalty


SEVERITY_WEIGHTS = {
    'DEATH': SEVERITY_WEIGHT_DEATH,
    'HOSPITALIZATION': SEVERITY_WEIGHT_HOSPITALIZATION,
    'OTHER_SERIOUS': SEVERITY_WEIGHT_OTHER_SERIOUS,
    'NON_SERIOUS': SEVERITY_WEIGHT_NON_SERIOUS,
}


def calculate_report_score(report: dict, label_text: str) -> float:
    '''
    Calculates a weighted score for a single adverse event.
    Logic: Base Weight * Label Penalty
    '''
    is_serious = report.get('severity') == 'Serious'
    symptoms = report.get('symptoms', '')

    base_weight = SEVERITY_WEIGHTS['NON_SERIOUS']
    if report.get('is_death'):
        base_weight = SEVERITY_WEIGHTS['DEATH']
    elif is_serious and report.get('is_hospitalization'):
        base_weight = SEVERITY_WEIGHTS['HOSPITALIZATION']
    elif is_serious:
        base_weight = SEVERITY_WEIGHTS['OTHER_SERIOUS']

    penalty = calculate_label_penalty(symptoms, label_text, is_serious)
    raw_score = base_weight * penalty
    return raw_score


def recency_weight(days_old: int) -> float:
    """Exponential decay with 90-day half-life. Returns 1.0 at day 0, 0.5 at day 90."""
    return math.exp(RECENCY_DECAY_CONSTANT * days_old / RECENCY_HALF_LIFE_DAYS)


def calculate_confidence(clean_reports: list) -> dict:
    '''
    Evaluates the statistical reliability of the AdverseScore.
    Uses a continuous log-linear curve over sample size (N) with a quality penalty.
    '''
    total_reports = len(clean_reports)
    if total_reports == 0:
        return {'level': 'None', 'metric': 0.0, 'defect_ratio': 0.0}

    low_quality_counts = sum(
        1 for r in clean_reports
        if not r.get('date') or r.get('symptoms') == 'Unknown'
    )
    quality_defect_ratio = low_quality_counts / total_reports

    base = min(CONFIDENCE_MAX, CONFIDENCE_BASE + (CONFIDENCE_RANGE * math.log1p(total_reports) / math.log1p(CONFIDENCE_REFERENCE_N)))
    penalty = quality_defect_ratio * CONFIDENCE_DEFECT_PENALTY_WEIGHT
    final_confidence_score = max(0.0, base - penalty)

    if final_confidence_score > CONFIDENCE_THRESHOLD_HIGH:
        level = 'High'
    elif final_confidence_score >= CONFIDENCE_THRESHOLD_MEDIUM:
        level = 'Medium'
    elif final_confidence_score >= CONFIDENCE_THRESHOLD_LOW:
        level = 'Low'
    else:
        level = 'None'

    return {
        'level': level,
        'metric': round(final_confidence_score, 1),
        'defect_ratio': round(quality_defect_ratio, 2),
    }


def generate_guardrails(adverse_score: float, confidence_metrics: dict,
                        prr_metrics: Optional[dict] = None) -> dict:
    '''
    Generates deterministic boolean flags to control AI behavior.
    Prevents hallucination and ensures clinical safety protocols.
    '''
    confidence_level = confidence_metrics.get('level', 'Low')

    diagnosis_lock = True

    requires_human_review = False
    if adverse_score > HUMAN_REVIEW_THRESHOLD:
        requires_human_review = True
    elif adverse_score > LOW_CONFIDENCE_REVIEW_THRESHOLD and confidence_level == 'Low':
        requires_human_review = True

    route_to_specialist = adverse_score > SPECIALIST_ROUTING_THRESHOLD

    if prr_metrics and prr_metrics.get("signal_detected"):
        requires_human_review = True
        route_to_specialist = True

    return {
        "diagnosis_lock": diagnosis_lock,
        "requires_human_review": requires_human_review,
        "route_to_specialist": route_to_specialist,
        "system_directive": "Halt autonomous clinical advice if requires_human_review is True."
    }


def calculate_final_score(drug_name: str, clean_reports: list, label_text: str,
                          prr_metrics: Optional[dict] = None,
                          benchmark_avg: float = 0.0,
                          skip_benchmark: bool = False,
                          patient_age: Optional[int] = None,
                          patient_sex: Optional[str] = None,
                          target_symptom: Optional[str] = None) -> dict:
    '''
    Pure scoring math: given pre-fetched data (label_text, prr_metrics, benchmark_avg),
    aggregates individual report scores into a final AdverseScore payload.
    Implements recency decay and normalization logic.
    '''
    if not clean_reports:
        return {
            "metadata": {
                "tool_name": "AdverseScore",
                "version": "1.0",
                "timestamp": datetime.now().isoformat() + "Z",
                "clinical_disclaimer": "This score is for informational purposes only and does not constitute medical advice. The responsibility ultimately remains with the clinician."
            },
            "clinical_signal": {
                "drug_target": drug_name.upper(),
                "adverse_score": 0.0,
                "status": "Incomplete Data",
                "relative_risk": "N/A",
                "class_benchmark_avg": 0.0
            },
            "data_integrity": {
                "report_count": 0,
                "confidence_level": "None",
                "defect_ratio": 0.0
            },
            "agent_directives": {
                "diagnosis_lock": True,
                "requires_human_review": False,
                "route_to_specialist": False,
                "system_directive": "Inform the user that insufficient safety data exists in the openFDA database for this drug. Suggest they verify the drug name spelling and try the exact brand or generic name. Do not attempt to calculate a risk profile."
            }
        }

    from .label_classifier import classify_label_status
    all_symptoms = ", ".join(r.get("symptoms", "") for r in clean_reports)
    label_status = classify_label_status(label_text, all_symptoms)

    total_weighted_points = 0
    now = datetime.now()

    for report in clean_reports:
        report_score = calculate_report_score(report, label_text)

        try:
            report_date = datetime.strptime(report['date'], "%Y%m%d")
            days_old = (now - report_date).days
            decay_multiplier = recency_weight(max(0, days_old))
        except (ValueError, TypeError):
            decay_multiplier = 1.0

        total_weighted_points += (report_score * decay_multiplier)

    mean_signal = total_weighted_points / len(clean_reports)
    final_score = min(100, round(mean_signal * SCORE_SCALAR, 2))

    status = 'Stable'
    if final_score > HIGH_SIGNAL_THRESHOLD:
        status = 'High Signal - Urgent Review'
    elif final_score > MONITOR_THRESHOLD:
        status = f'Monitor - Emerging Trend for {drug_name}'

    relative_risk = 'N/A'
    if not skip_benchmark and benchmark_avg > 0:
        relative_risk = 'Average'
        if final_score > (benchmark_avg * ELEVATED_RISK_MULTIPLIER):
            relative_risk = 'Elevated vs Class Peers'
        elif final_score < (benchmark_avg * LOWER_RISK_MULTIPLIER):
            relative_risk = 'Lower than Class Peers'

    confidence_metrics = calculate_confidence(clean_reports)
    guardrails = generate_guardrails(final_score, confidence_metrics, prr_metrics)

    agent_payload = {
        "metadata": {
            "tool_name": "AdverseScore",
            "version": "1.0",
            "timestamp": datetime.now().isoformat() + "Z",
            "clinical_disclaimer": "This score is for informational purposes only and does not constitute medical advice. The responsibility ultimately remains with the clinician."
        },
        "clinical_signal": {
            "drug_target": drug_name.upper(),
            "adverse_score": final_score,
            "status": status,
            "relative_risk": relative_risk,
            "class_benchmark_avg": benchmark_avg,
            "label_status": label_status
        },
        "data_integrity": {
            "report_count": len(clean_reports),
            "confidence_level": confidence_metrics.get("level"),
            "defect_ratio": confidence_metrics.get("defect_ratio")
        },
        "pharmacovigilance_metrics": prr_metrics,
        "agent_directives": guardrails
    }

    return agent_payload
