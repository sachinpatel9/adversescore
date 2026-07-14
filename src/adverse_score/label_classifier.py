from dataclasses import dataclass


def classify_label_status(label_text: str, symptoms_str: str) -> str:
    '''
    Classifies a symptom string as LABELED, UNLABELED, or LABEL_STATUS_UNKNOWN
    based on whether any symptom appears in the drug's official FDA label text.
    '''
    if not label_text:
        return "LABEL_STATUS_UNKNOWN"
    symptom_list = [s.strip().lower() for s in symptoms_str.split(",") if s.strip()]
    if not symptom_list:
        return "LABEL_STATUS_UNKNOWN"
    if any(s in label_text for s in symptom_list):
        return "LABELED"
    return "UNLABELED"


@dataclass(frozen=True)
class LabelClassificationResult:
    statuses: dict          # {symptom_upper: "LABELED"|"UNLABELED"|"LABEL_STATUS_UNKNOWN"}
    total_symptoms: int
    labeled_count: int
    unlabeled_count: int
    unknown_count: int


def classify_label_statuses(label_text: str, symptoms: list) -> LabelClassificationResult:
    """
    Batch version of classify_label_status, for Phase 5 ranking consumption: classifies
    every unique symptom in `symptoms` against the same label_text, keyed by normalized
    (upper-cased) symptom — matches the uppercasing convention already used for symptom
    keys in prr.py's output (`target_symptom.upper()`) and deduplication.py's heuristic
    matching (`.strip().upper()` on symptom_list).

    Reuses classify_label_status() per unique symptom rather than re-implementing the
    substring-match logic. Handles an empty symptoms list and empty label_text gracefully
    (no exceptions); every symptom maps to LABEL_STATUS_UNKNOWN when label_text is empty,
    which falls out naturally from the underlying function.
    """
    normalized_symptoms = sorted({
        s.strip().upper() for s in symptoms if s and s.strip()
    })

    statuses = {}
    labeled_count = 0
    unlabeled_count = 0
    unknown_count = 0
    for symptom in normalized_symptoms:
        status = classify_label_status(label_text, symptom)
        statuses[symptom] = status
        if status == "LABELED":
            labeled_count += 1
        elif status == "UNLABELED":
            unlabeled_count += 1
        else:
            unknown_count += 1

    return LabelClassificationResult(
        statuses=statuses,
        total_symptoms=len(normalized_symptoms),
        labeled_count=labeled_count,
        unlabeled_count=unlabeled_count,
        unknown_count=unknown_count,
    )
