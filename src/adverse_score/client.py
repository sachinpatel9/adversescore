from typing import Optional
from .fda_client import FDAClient
from .label_classifier import calculate_label_penalty, classify_label_status
from .prr import calculate_prr
from .scoring import (calculate_final_score as _compute_score,
                      calculate_report_score, calculate_confidence,
                      generate_guardrails, SEVERITY_WEIGHTS)


class AdverseScoreClient:
    """Thin orchestrator that coordinates FDA data retrieval with scoring math."""

    base_url = "https://api.fda.gov/drug/event.json"
    SEVERITY_WEIGHTS = SEVERITY_WEIGHTS

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

    # ── Pure function delegation (keeps test compat) ────────────────────
    def calculate_label_penalty(self, *a, **kw):     return calculate_label_penalty(*a, **kw)
    def _classify_label_status(self, *a, **kw):      return classify_label_status(*a, **kw)
    def _calculate_report_score(self, *a, **kw):     return calculate_report_score(*a, **kw)
    def _calculate_confidence(self, *a, **kw):       return calculate_confidence(*a, **kw)
    def _generate_guardrails(self, *a, **kw):        return generate_guardrails(*a, **kw)

    # ── Orchestration methods ───────────────────────────────────────────
    def calculate_final_score(self, drug_name: str, clean_reports: list,
                              skip_benchmark: bool = False,
                              patient_age: Optional[int] = None,
                              patient_sex: Optional[str] = None,
                              target_symptom: Optional[str] = None) -> dict:
        """Gathers label text, PRR metrics, and benchmark data, then delegates to scoring math."""
        if not clean_reports:
            return _compute_score(drug_name, [], "")

        label_text = self.fetch_label_text(drug_name)

        prr_metrics = None
        if target_symptom:
            pharm_class = self._discover_drug_class(drug_name)
            if pharm_class:
                prr_metrics = self._calculate_prr_metrics(
                    drug_name, pharm_class, target_symptom,
                    patient_age, patient_sex, label_text=label_text)

        benchmark_avg = 0.0
        if not skip_benchmark:
            benchmark_avg = self.get_peer_benchmark(drug_name, patient_age, patient_sex)

        return _compute_score(
            drug_name, clean_reports, label_text,
            prr_metrics=prr_metrics, benchmark_avg=benchmark_avg,
            skip_benchmark=skip_benchmark,
            patient_age=patient_age, patient_sex=patient_sex,
            target_symptom=target_symptom)

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

    def get_peer_benchmark(self, drug_name: str,
                           patient_age: Optional[int] = None,
                           patient_sex: Optional[str] = None) -> float:
        """Calculates the average AdverseScore of peer drugs using Dynamic Ontology Mapping."""
        target_upper = drug_name.upper()
        pharm_class = self._discover_drug_class(target_upper)
        peers = self._discover_peers(pharm_class, target_upper)

        if not peers:
            print(f'[Benchmarking] No peers discovered for {drug_name}. Skipping benchmark.')
            return 0.0

        peer_scores = []
        print(f"[Benchmarking] Evaluating {drug_name} against the discovered peers: {', '.join(peers)}")

        for peer in peers:
            print(f'[Benchmarking] Fetching clinical data for {peer}...')
            raw = self.fetch_events(peer, patient_age, patient_sex)
            clean = self._flatten_results(raw)
            if not clean:
                print(f'[Benchmarking] No data for peer {peer}, excluding from benchmark.')
                continue
            result = self.calculate_final_score(peer, clean, skip_benchmark=True)
            peer_score = result['clinical_signal']['adverse_score']
            peer_scores.append(peer_score)

        return round(sum(peer_scores) / len(peer_scores), 2) if peer_scores else 0.0

    def fetch_quarterly_data(self, drug_name, num_quarters=4,
                             patient_age=None, patient_sex=None,
                             target_symptom=None):
        """Calculate AdverseScore and PRR per quarter. Returns [{quarter, adverse_score, prr, report_count}]."""
        boundaries = self._compute_quarter_boundaries(num_quarters)
        pharm_class = None
        if target_symptom:
            pharm_class = self._discover_drug_class(drug_name)

        time_series = []
        for label, start, end in boundaries:
            raw = self.fetch_events(drug_name, patient_age=patient_age,
                                    patient_sex=patient_sex, start_date=start, end_date=end)
            reports = self._flatten_results(raw) if raw else []
            report_count = len(reports)

            if reports:
                result = self.calculate_final_score(
                    drug_name, reports, skip_benchmark=True,
                    patient_age=patient_age, patient_sex=patient_sex,
                    target_symptom=target_symptom)
                score = result["clinical_signal"]["adverse_score"]
            else:
                score = 0

            prr_value = None
            if target_symptom and pharm_class:
                drug_counts = self._fetch_symptom_counts(
                    drug_name=drug_name, patient_age=patient_age,
                    patient_sex=patient_sex, start_date=start, end_date=end)
                class_counts = self._fetch_symptom_counts(
                    pharm_class=pharm_class, patient_age=patient_age,
                    patient_sex=patient_sex, start_date=start, end_date=end)
                a = drug_counts.get(target_symptom.upper(), 0)
                a_plus_b = sum(drug_counts.values()) or 1
                c = class_counts.get(target_symptom.upper(), 0)
                c_plus_d = sum(class_counts.values()) or 1
                if a >= 3 and c > 0 and c_plus_d > 0 and a_plus_b > 0:
                    prr_value = round((a / a_plus_b) / (c / c_plus_d), 2)

            time_series.append({
                "quarter": label,
                "adverse_score": round(score),
                "prr": prr_value,
                "report_count": report_count,
            })
        return time_series

    def compute_trend(self, time_series):
        """Classify trend: RISING/STABLE/DECLINING/INSUFFICIENT_DATA."""
        valid = [q for q in time_series if q.get("report_count", 0) > 0]
        if len(valid) < 2:
            return "INSUFFICIENT_DATA"
        recent = valid[-1]["adverse_score"]
        comparison = valid[-3]["adverse_score"] if len(valid) >= 3 else valid[0]["adverse_score"]
        delta = recent - comparison
        if delta >= 10:
            return "RISING"
        elif delta <= -10:
            return "DECLINING"
        return "STABLE"
