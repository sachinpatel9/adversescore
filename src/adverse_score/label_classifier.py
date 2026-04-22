from .config import (
    LABEL_PENALTY_UNLABELED_SERIOUS, LABEL_PENALTY_UNLABELED_NON_SERIOUS,
    LABEL_PENALTY_LABELED,
)


def calculate_label_penalty(symptoms: str, label_text: str, is_serious: bool) -> float:
    '''
    Apply the penalty factors defined.
    Unlabeled + Serious: 2.0x | Unlabeled + Non-Serious: 1.5x | Labeled: 1.0x
    '''
    if not label_text:
        return LABEL_PENALTY_UNLABELED_SERIOUS if is_serious else LABEL_PENALTY_UNLABELED_NON_SERIOUS

    symptom_list = [s.strip().lower() for s in symptoms.split(",") if s.strip()]
    if not symptom_list:
        return LABEL_PENALTY_UNLABELED_SERIOUS if is_serious else LABEL_PENALTY_UNLABELED_NON_SERIOUS
    is_labeled = any(s in label_text for s in symptom_list)

    if not is_labeled:
        return LABEL_PENALTY_UNLABELED_SERIOUS if is_serious else LABEL_PENALTY_UNLABELED_NON_SERIOUS
    return LABEL_PENALTY_LABELED


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
