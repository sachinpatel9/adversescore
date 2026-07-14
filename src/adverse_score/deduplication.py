from dataclasses import dataclass
from .logger import get_logger, log_event

logger = get_logger("deduplication")


@dataclass(frozen=True)
class DedupResult:
    reports: list                      # deduplicated flattened report dicts
    total_input_count: int
    total_output_count: int
    removed_by_exact_id: int
    removed_by_heuristic: int
    audit_trail: list                  # [{"removed_id": ..., "kept_id": ..., "method": "exact_id"|"heuristic"}, ...]


def deduplicate_reports(reports: list) -> DedupResult:
    """
    Removes duplicate FAERS case reports from a flattened report list (the shape
    produced by fda_client.py's _flatten_results / PSURRetrievalResult.reports).

    Two passes:
    1. Exact safetyreportid match — keeps the highest safetyreportversion per id.
       This is defensive/idempotent: fetch_psur_reports() already runs a
       version-aware _merge_chunks() pass, so in the normal pipeline this pass
       should find nothing left to do. It exists so deduplicate_reports() is
       correct as a standalone entry point on any list[dict] source, not just
       Phase 2's output.
    2. Fallback heuristic match on Pass 1 survivors — reports with a DIFFERENT
       safetyreportid are treated as the same real-world case if their full
       normalized drug_names set AND full normalized symptom_list set AND date
       are all identical (full-set equality, not "any shared term" — chosen to
       avoid over-merging genuinely distinct multi-symptom cases).

    LIMITATIONS (documented, not silently resolved — matches the scope doc's
    Section 7 style):
    - `date` here is `receivedate` (FDA receipt date), not a true adverse-event
      onset date; FAERS has no single unambiguous per-reaction event date, and
      this is the same proxy already used elsewhere in this codebase.
    - Reports with an empty drug_names or empty symptom_list are excluded from
      heuristic grouping entirely (never merged this way) — an empty set matching
      another empty set would be a meaningless, over-eager merge.
    """
    total_input_count = len(reports)
    audit_trail = []

    # Pass 1: exact safetyreportid match, keep-highest-version tiebreak.
    # A missing/None report_id carries no exact-ID signal at all and must never be
    # grouped against another missing-id report (would silently collapse distinct
    # cases that merely both lack an ID) — each gets a unique synthetic key so it
    # always survives Pass 1 untouched.
    best_by_key = {}
    order = []
    for report in reports:
        rid = report.get("report_id")
        key = rid if rid is not None else object()
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = report
            order.append(key)
        else:
            incoming_version = report.get("safetyreportversion", 1)
            existing_version = existing.get("safetyreportversion", 1)
            if incoming_version > existing_version:
                audit_trail.append({"removed_id": existing.get("report_id"),
                                     "kept_id": report.get("report_id"), "method": "exact_id"})
                best_by_key[key] = report
            else:
                audit_trail.append({"removed_id": report.get("report_id"),
                                     "kept_id": existing.get("report_id"), "method": "exact_id"})
    pass1_survivors = [best_by_key[key] for key in order]
    removed_by_exact_id = total_input_count - len(pass1_survivors)

    # Pass 2: fallback heuristic on Pass 1 survivors only.
    def _heuristic_key(report):
        drug_names = report.get("drug_names") or []
        symptom_list = report.get("symptom_list") or []
        drug_key = frozenset(n.strip().upper() for n in drug_names if n and n.strip())
        symptom_key = frozenset(t.strip().upper() for t in symptom_list if t and t.strip())
        return (drug_key, symptom_key, report.get("date"))

    seen_keys = {}
    final_reports = []
    removed_by_heuristic = 0
    for report in pass1_survivors:
        drug_names = report.get("drug_names") or []
        symptom_list = report.get("symptom_list") or []
        if not drug_names or not symptom_list:
            # Can't meaningfully compare — treat as unique, never heuristically merged.
            final_reports.append(report)
            continue
        key = _heuristic_key(report)
        if key in seen_keys:
            audit_trail.append({"removed_id": report.get("report_id"),
                                 "kept_id": seen_keys[key], "method": "heuristic"})
            removed_by_heuristic += 1
            continue
        seen_keys[key] = report.get("report_id")
        final_reports.append(report)

    total_output_count = len(final_reports)
    if removed_by_exact_id or removed_by_heuristic:
        log_event(logger, "deduplication_complete", total_input=total_input_count,
                  total_output=total_output_count, removed_exact_id=removed_by_exact_id,
                  removed_heuristic=removed_by_heuristic)

    return DedupResult(
        reports=final_reports,
        total_input_count=total_input_count,
        total_output_count=total_output_count,
        removed_by_exact_id=removed_by_exact_id,
        removed_by_heuristic=removed_by_heuristic,
        audit_trail=audit_trail,
    )
