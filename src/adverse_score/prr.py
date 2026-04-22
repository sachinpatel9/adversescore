import math
from .label_classifier import classify_label_status


def calculate_prr(drug_counts: dict, class_counts: dict,
                  target_symptom: str, label_text: str = "") -> dict:
    '''
    Pure PRR + Wald 95% CI math. Given pre-fetched symptom count dicts,
    computes the Proportional Reporting Ratio and confidence interval.

    METHODOLOGY NOTE (pharmacovigilance deviation from Evans et al., 2001):
    1. The classical PRR comparator is "all OTHER drugs" (excluding the target).
       This implementation uses "all drugs in the same pharmacologic class" which
       INCLUDES the target drug in the denominator. When the class has many drugs
       the impact is small, but for small classes this dilutes the signal.
    2. The denominators (a+b, c+d) are total symptom MENTION counts from the
       openFDA count endpoint, not total REPORT counts. Since a single report can
       list multiple symptoms, this inflates the denominators and slightly deflates
       the PRR vs. the classical report-level formulation.
    Both are accepted simplifications in automated signal detection systems where
    individual report-level 2x2 tables are impractical to construct from the API.
    '''
    symptom_upper = target_symptom.upper()

    # a = target_drug + target_symptom
    # a_plus_b = Total Target Drug Symptoms
    a = drug_counts.get(symptom_upper, 0)
    a_plus_b = sum(drug_counts.values())

    # c = class + target symptom
    # c_plus_d = total class symptoms
    c = class_counts.get(symptom_upper, 0)
    c_plus_d = sum(class_counts.values())

    # classify label status for the target symptom
    symptom_label_status = classify_label_status(label_text, target_symptom)

    # guard against division by zero or stat insignificant sample size
    if a < 3 or c == 0 or a_plus_b == 0 or c_plus_d == 0:
        return {"prr": 0.0, "ci_lower": 0.0, "signal_detected": False,
                "target_symptom": symptom_upper, "drug_cases": a,
                "class_cases": c, "label_status": symptom_label_status}

    prr = (a / a_plus_b) / (c / c_plus_d)

    try:
        # 95% CI lower bound for log-transformed PRR (Wald method).
        # SE(ln(PRR)) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
        se = math.sqrt((1/a) + (1/c) - (1/a_plus_b) - (1/c_plus_d))
        ci_lower = math.exp(math.log(prr) - 1.96 * se)
    except ValueError:
        ci_lower = 0.0

    # Mathematic Guardrail
    signal_detected = ci_lower > 1.0 and a >= 3

    if signal_detected:
        print(f"[Math Engine] Stat Signal Detected: {symptom_upper} | PRR: {round(prr,2)} | CI Lower: {round(ci_lower, 2)}")

    return {
        "prr": round(prr, 2),
        "ci_lower": round(ci_lower, 2),
        "signal_detected": signal_detected,
        "target_symptom": target_symptom,
        "drug_cases": a,
        "class_cases": c,
        "label_status": symptom_label_status
    }
