def calculate_label_penalty(symptoms: str, label_text: str, is_serious: bool) -> float:
    '''
    Apply the penalty factors defined.
    Unlabeled + Serious: 2.0x | Unlabeled + Non-Serious: 1.5x | Labeled: 1.0x
    '''
    if not label_text:
        return 2.0 if is_serious else 1.5

    symptom_list = [s.strip().lower() for s in symptoms.split(",") if s.strip()]
    if not symptom_list:
        return 2.0 if is_serious else 1.5
    is_labeled = any(s in label_text for s in symptom_list)

    if not is_labeled:
        return 2.0 if is_serious else 1.5
    return 1.0


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
