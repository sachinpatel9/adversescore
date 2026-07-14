from typing import Optional
from .fda_client import FDAClient
from .label_classifier import classify_label_status
from .logger import get_logger, log_event
from .prr import calculate_prr

logger = get_logger("client")


class AdverseScoreClient:
    """Thin orchestrator that coordinates FDA data retrieval with pure math modules."""

    base_url = "https://api.fda.gov/drug/event.json"

    def __init__(self):
        self.fda = FDAClient()
        self.api_key = self.fda.api_key
        self.session = self.fda.session  # backward compat for monkeypatches

    # ── FDA delegation (keeps test/agent_tools compat) ──────────────────
    def build_query(self, *a, **kw):                return self.fda.build_query(*a, **kw)
    def fetch_events(self, *a, **kw):               return self.fda.fetch_events(*a, **kw)
    def _flatten_results(self, *a, **kw):            return self.fda._flatten_results(*a, **kw)
    def fetch_label_text(self, *a, **kw):            return self.fda.fetch_label_text(*a, **kw)
    def _discover_drug_class(self, *a, **kw):        return self.fda._discover_drug_class(*a, **kw)
    def _fetch_label_class_fallback(self, *a, **kw): return self.fda._fetch_label_class_fallback(*a, **kw)
    def _discover_peers(self, *a, **kw):             return self.fda._discover_peers(*a, **kw)
    def _fetch_symptom_counts(self, *a, **kw):       return self.fda._fetch_symptom_counts(*a, **kw)
    def _sanitize_for_query(self, *a, **kw):         return self.fda._sanitize_for_query(*a, **kw)
    def _compute_quarter_boundaries(self, *a, **kw): return self.fda._compute_quarter_boundaries(*a, **kw)
    def _get_transport_session(self):                return self.fda._get_transport_session()
    def _resilient_get(self, *a, **kw):             return self.fda._resilient_get(*a, **kw)

    # ── Pure function delegation (keeps test compat) ────────────────────
    def _classify_label_status(self, *a, **kw):      return classify_label_status(*a, **kw)

    # ── Orchestration methods ───────────────────────────────────────────
    def _calculate_prr_metrics(self, drug_name: str,
                               pharm_class: Optional[str] = None,
                               target_symptom: str = "",
                               patient_age: Optional[int] = None,
                               patient_sex: Optional[str] = None,
                               label_text: str = "",
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> dict:
        """Fetches symptom counts from FDA, then delegates to pure PRR math."""
        if not pharm_class:
            pharm_class = self._discover_drug_class(drug_name)
        if not pharm_class:
            return {"prr": 0.0, "ci_lower": 0.0, "signal_detected": False,
                    "target_symptom": target_symptom.upper(), "drug_cases": 0,
                    "class_cases": 0, "label_status": "LABEL_STATUS_UNKNOWN"}
        drug_counts = self._fetch_symptom_counts(
            drug_name=drug_name, patient_age=patient_age,
            patient_sex=patient_sex, start_date=start_date, end_date=end_date)
        class_counts = self._fetch_symptom_counts(
            pharm_class=pharm_class, patient_age=patient_age,
            patient_sex=patient_sex, start_date=start_date, end_date=end_date)
        return calculate_prr(drug_counts, class_counts, target_symptom, label_text)
