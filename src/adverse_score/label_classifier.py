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
