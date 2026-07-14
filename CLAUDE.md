# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdverseScore is being rebuilt as a **PSUR consolidation agent** for pharmacovigilance teams. Given a drug name and reporting period, it consolidates FAERS adverse event data, deduplicates case reports, ranks signals deterministically, and generates structured `.docx` documents aligned to ICH E2C(R2) PBRER format. The rebuild is tracked in the authoritative scope document `docs/PSUR_CONSOLIDATION_SCOPE.md` (11-phase plan). **Currently completed:** Phase 0 (Foundation Cleanup), Phase 1 (Drug Identity Resolution), Phase 2 (Chunked, Paginated FDA Retrieval), Phase 3 (Deduplication), Phase 4 (Label Status Classification), and Phase 5 (Deterministic Ranking Engine). Phases 6–11 (document generation, new UI, agent orchestration) are in progress.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run app.py
```

Requires a `.env` file with `OPENFDA_API_KEY` and `OPENAI_API_KEY` (see `.env.example`).

## Architecture

Current state (Phase 0–5 complete): Foundation cleanup, drug identity resolution, chunked FDA retrieval, deduplication, label status classification, and deterministic ranking engine are done. Phase 6+ modules (document generation, new agent orchestration) are planned but not yet built.

- **`app.py`** — Streamlit UI. Currently a placeholder chat interface (shows "PSUR consolidation agent is being rebuilt" message instead of invoking a live agent). Will be rebuilt in Phases 8–10 to display PSUR period selector, ranked signal list, and `.docx` export controls.
- **`src/adverse_score/drug_identity.py`** (NEW, Phase 1) — Given a raw drug name, resolves it to canonical brand/generic name variants and market authorization/approval date. Entry point: `resolve_drug_identity(raw_name: str, client: Optional[FDAClient] = None)`. Uses openFDA label/NDC queries (exact then broadened) followed by `drugsfda.json` for approval date lookup. Returns `DrugIdentity` dataclass (with resolution_confidence) or `DrugIdentityError` on failure. Known limitation: OTC monograph drugs (aspirin, ibuprofen) have no `drugsfda.json` entry, so approval_date is `None` (marked `PARTIAL` confidence, not an error).
- **`src/adverse_score/client.py`** — Much-reduced orchestrator. Removed: `calculate_final_score`, `get_peer_benchmark`, `fetch_quarterly_data`, `compute_trend`, all scoring-related methods. Retained: FDA delegation one-liners (`fetch_events`, `fetch_label_text`, `build_query`, etc.), `_classify_label_status`, `_calculate_prr_metrics`. No longer requires API keys at construction time (backward-compatible skeleton pattern).
- **`src/adverse_score/fda_client.py`** — openFDA HTTP client (~500 lines). Existing methods: `fetch_events`, `fetch_label_text`, `_discover_drug_class`, `_discover_peers`, `_fetch_symptom_counts`, `build_query`, `_flatten_results`, `_compute_quarter_boundaries`. Dual-layer retry (urllib3 for transport codes, tenacity for app-level transience). Phase 2 additions: module-level pure functions `_add_months()` (stdlib day-of-month clamping), `compute_psur_period()` (anchors PSUR cycle to market_authorization_date, selecting most recently completed fixed-length cycle; falls back to rolling lookback for OTC or pre-approval drugs), `_compute_psur_chunks()` (slices PSUR period into quarterly chunks with labels `"chunk-{i}-YYYYMMDD-to-YYYYMMDD"`), `_merge_chunks()` (first-seen-wins dedup on report_id for boundary collisions). New `FDAClient` methods: `_build_psur_chunk_query()` (composes OR'd drug name variants + AND'd date range via params dict), `_fetch_chunk_paginated()` (paginates single chunk with skip/limit capped at `PSUR_SKIP_CEILING=25000`; aborts on per-chunk error without failing other chunks). New public method: `fetch_psur_reports()` (top-level entry point orchestrating all of the above; returns `PSURRetrievalResult` never raises except ValueError for empty name_variants list). Two new frozen dataclasses: `ChunkResult` (per-chunk metadata: label, dates, reports, counts, truncated flag, optional error) and `PSURRetrievalResult` (aggregate: period bounds, all chunks, merged reports, totals, completeness flags). Layering note: does NOT import `drug_identity.py` (avoids circular import); accepts plain primitives (name variants list, optional date, period string) instead. Phase 3 additions: `_flatten_results()` now emits three new keys on each flattened report dict (`symptom_list` as a list of MedDRA PT terms, `drug_names` as a deduped list of medicinal product names, `safetyreportversion` as int), and `_merge_chunks()` is now version-aware (when same `report_id` recurs across chunks, higher `safetyreportversion` wins instead of blind first-seen-wins). Two new module-level helpers: `_extract_drug_names(report)`, `_parse_version(raw)`. Phase 5 additions: `_flatten_results()` now also extracts per-report `reactions` key (`[{"term": <MedDRA PT>, "outcome_code": <int 1-6 or None>}, ...]`) from raw FAERS `patient.reaction[].reactionoutcome` field via new `_parse_outcome_code(raw)` helper (never raises; returns `None` for missing/malformed/out-of-range values, enabling safe reversibility tiering for Phase 5 ranking).
- **`src/adverse_score/deduplication.py`** (NEW, Phase 3) — Pure deduplication module (no HTTP, no `FDAClient` import), same architectural pattern as `prr.py`/`label_classifier.py`. Removes duplicate FAERS case reports from a flattened report list. Entry point: `deduplicate_reports(reports: list) -> DedupResult`. Two-pass implementation: (1) Exact `safetyreportid` match with version awareness (groups by `report_id`, keeps highest `safetyreportversion` per id; reports with missing/`None` `report_id` get unique synthetic keys and never collapse), (2) Fallback heuristic match on survivors only (two reports with different `report_ids` are duplicates only if their full normalized `drug_names` set AND full normalized `symptom_list` set AND `date` are ALL identical — deliberately strict to avoid over-merging). Reports with empty `drug_names` or empty `symptom_list` are excluded from heuristic matching. Returns `DedupResult` (frozen dataclass): `reports`, `total_input_count`, `total_output_count`, `removed_by_exact_id`, `removed_by_heuristic`, `audit_trail` (list of removal dicts). Known limitation: `date` here is `receivedate` (FDA receipt date), not a true adverse-event onset date—this is the same proxy used elsewhere in the codebase.
- **`src/adverse_score/ranking.py`** (NEW, Phase 5) — Deterministic signal ranking engine (no HTTP, no `FDAClient` import), pure module importing only `dataclasses`, `.config`, `.prr`, `.label_classifier`, `.logger`. Entry point: `rank_signals(reports: list, class_counts: dict, label_text: str) -> RankingResult`. A "signal" is one unique MedDRA PT symptom across deduplicated reports. Ranks each signal across four lexicographically-ordered tiers: (1) **Seriousness & Outcome** — DEATH > HOSPITALIZATION > OTHER_SERIOUS > NON_SERIOUS (derived from existing `is_death`/`is_hospitalization`/`severity` fields), (2) **Strength of Evidence** — STRONG_UNLABELED > STRONG_LABELED > WEAK_UNLABELED > WEAK_LABELED > UNKNOWN_LABEL_STATUS (reuses `prr.py`'s `calculate_prr()` output; unlabeled outranks labeled at equivalent evidence strength per scope doc), (3) **Reversibility** — FATAL > POOR > REVERSIBLE > UNKNOWN (derived from newly-extracted per-reaction FAERS outcome codes via Phase 5 `fda_client.py` changes; UNKNOWN used when outcome data is absent, never silently assumed reversible), (4) **Public Health Impact** — HIGH > MODERATE > LOW (derived from report-volume thresholds: 100/20). Returns `RankedSignal` (frozen dataclass: `symptom`, `rank`, `seriousness_tier`, `strength_of_evidence_tier`, `reversibility_tier`, `public_health_tier`, `prr_metrics` dict, `report_count`) and `RankingResult` (frozen dataclass: `ranked_signals` list, `label_summary` as `LabelClassificationResult`, `formula_version` string, `total_signals` int). Sort key is purely internal (tuple of tier ordinals); never stored/exposed on output dataclasses — this design explicitly avoids the banned single composite score. Known limitation: reversibility is a heuristic approximation based on outcome codes; true clinical reversibility assessment requires expert review.
- **`src/adverse_score/prr.py`** — Pure PRR + Wald 95% CI math (~70 lines). `calculate_prr(drug_counts, class_counts, target_symptom, label_text)`. Unchanged; will feed Phase 4+ ranking engine.
- **`src/adverse_score/label_classifier.py`** — Pure label classification. Contains `classify_label_status(label_text: str, symptoms_str: str) -> str` (single symptom, LABELED/UNLABELED/LABEL_STATUS_UNKNOWN via substring match; unchanged from prior phases, used by prr.py) and NEW Phase 4: `classify_label_statuses(label_text: str, symptoms: list) -> LabelClassificationResult` (batch variant for Phase 5 ranking consumption). The batch function normalizes symptoms via `.strip().upper()`, dedupes case-insensitively, filters empty entries, sorts for determinism, and calls the existing single-symptom function once per unique symptom. Returns frozen dataclass `LabelClassificationResult` with per-symptom status dict, total/labeled/unlabeled/unknown counts. Invariant: `labeled_count + unlabeled_count + unknown_count == total_symptoms`.
- **`src/adverse_score/orchestrator.py`** — Placeholder. `agent_executor = None` sentinel; importing no longer requires API keys. Will be rebuilt in Phase 8 with new LangGraph agent, multi-turn guidance, and 7-guardrail system prompt.
- **`src/adverse_score/persistence.py`** — Reduced to bare skeleton: `__init__`, no-op `_init_schema`, context-manager protocol. Old `AnalysisStore.analyses` table and CRUD methods removed. Phase 7 will design new schema for consolidated datasets + conversation history.
- **`src/adverse_score/logger.py`** — JSON-structured logging to stderr. Unchanged.
- **`src/adverse_score/config.py`** — Removed 26 old composite-score constants (severity weights, label penalties, recency decay, confidence curve, guardrail thresholds, etc.). Retained: `initialize_config()`, PRR constants, API timeout/retry constants, drug-class/peer-discovery constants. Added Phase 1 constants: `DRUGSFDA_ENDPOINT`, `NDC_ENDPOINT`, `DRUG_IDENTITY_*_LIMIT`, `DRUGSFDA_APPROVED_STATUS`. Added Phase 2 constants (PSUR Chunked Retrieval): `PSUR_PAGE_SIZE=1000`, `PSUR_SKIP_CEILING=25000`, `PSUR_CHUNK_MONTHS=3`, `PSUR_PERIOD_MONTHS` (dict: period string → month counts), `PSUR_PERIOD_FALLBACK_DAYS` (dict: period string → fallback day counts, used only on anchor-less path). Added Phase 5 constants (Deterministic Ranking Engine): `RANKING_FORMULA_VERSION` (current: "1.0"), seriousness/strength-of-evidence/reversibility/public-health tier name constants plus their fixed priority-order tuples, `REACTION_OUTCOME_*` FAERS outcome code constants (1-6), public-health volume thresholds (100 for HIGH, 20 for MODERATE, remaining as LOW).

## Workflow Multi-Agent Orchestration

Each phase's implementation plan must declare a "Critical Files" list (files that may be
created/modified) before any subagent is dispatched. All three subagents below operate
strictly within that declared file boundary unless a subagent discovers a genuine
cross-cutting bug — in which case it stops and reports back rather than silently expanding
scope.

**Dispatch mechanics:** All three subagents are dispatched via the `Agent` tool with
`subagent_type: general-purpose` (or `claude`). Never dispatch them as `Explore` or `Plan`
— those two agent types do not have Write/Edit tools and would silently break the full
tool access every subagent below requires.

1. Orchestrator Agent 
* Model: Fable 5 (OR Opus 4.8 when Fable 5 is no longer available due to usage limits)
* Tool Access: Full tool access (Read, Write, Edit, Bash, etc)
* Purpose: Plans and delegates to the SubAgent workers

2. Build SubAgent:
* Model: Sonnet 5 (high effort)
* Tool access: full tool access (Read, Write, Edit, Bash, etc.) — needed to write code and
  run tests directly.
* Scope: Implements the code + tests described in the phase's plan, touching only the files
  in that phase's "Critical Files" list.
* Out of scope: documentation files (CLAUDE.md, README.md), modules outside the declared
  file list, and already-passing tests unrelated to the phase's own changes.
* Produces: working code + a passing test suite for the declared scope, plus a concise
  summary of what was built (function/class names, file paths) for the Review SubAgent.

3. Review SubAgent:
* Model: attempt `model: sonnet` (Sonnet 5)
* Tool access: full tool access (Read, Write, Edit, Bash, etc.) — needed to make direct
  fixes to the code if bugs are identified.
* Scope: Reviews only the diff produced by the Build SubAgent, against the originating
  phase plan's acceptance criteria and test plan — not a general-purpose codebase audit.
  Acts as a skeptical Staff Developer: checks correctness, security invariants
  (`_sanitize_for_query()`, `params=` transport conventions), layering invariants, and test
  coverage against the plan's stated test cases. Uses this repo's project-scoped
  `staff-python-reviewer` skill (`.claude/skills/staff-python-reviewer/SKILL.md`), which
  carries the same staff-level review persona as the global skill plus AdverseScore-specific
  invariant checks.
* Fixes bugs it finds directly, within the same file boundary the Build SubAgent used —
  does not expand into unrelated refactors.
* Out of scope: documentation files. Runs only after the Build SubAgent reports completion.
* Produces: a short report of what was found/fixed, handed to the Write SubAgent.

4. Write SubAgent:
* Model: Haiku (Medium Effort)
* Tool access: full tool access (Read, Write, Edit, Bash, etc.), though its scope below
  only requires Read/Edit on CLAUDE.md and README.md.
* Scope: Updates CLAUDE.md and README.md ONLY, using the Build/Review summary as its
  factual source — never re-derives architecture independently and never touches code or
  tests.
* Runs only after the Review SubAgent confirms the phase's acceptance criteria are met
  (tests green).
* The documentation follows best industry practices: thorough in content, concise in
  structure, professional in tone.

NOTE: The goal of the Multi-Agent Orchestration is to optimize token usage and prevent
scope creep or overlapping edits between subagents — each implementation plan should
declare its Critical Files list explicitly so every subagent knows its boundary before
starting. ALL tools are available for each subagent to use for any purposes necessary; the
per-subagent "Tool access" lines above exist so file-boundary scope is never confused with
tool-access restriction — they are independent constraints.




## Key Design Decisions

- All query-building methods must call `_sanitize_for_query()` before embedding values in Lucene strings. This is a security invariant — check it when adding new FDA queries.
- openFDA sex codes: **1=Male, 2=Female**. This was previously inverted and is a common source of bugs.
- A pre-commit hook in `.git/hooks/pre-commit` blocks `.env` files and scans for API key patterns.
- **URL encoding in `fda_client.py`**: Query parameters must be passed via the `params=` dict to `session.get()`. Do NOT use `urllib.parse.quote()` manually — the `requests` library handles encoding. Critically: when composing multi-term Lucene queries with "OR"/"AND" keywords, use literal spaces (e.g., `"KEYTRUDA OR OPDIVO"`) in the param value; using literal "+OR+" text gets double-encoded to `%2B` by `requests`, breaking the query server-side. Drug/class names themselves are still routed through `_sanitize_for_query()` before being embedded in Lucene field strings.
- **`_calculate_prr_metrics`** accepts optional `start_date` and `end_date` parameters for time-bounded PRR calculation. Pass these through to `_fetch_symptom_counts` when computing per-quarter or per-period analysis.
- **Dual-layer HTTP retry**: `fda_client.py` uses urllib3 `Retry` for transport-level status code retries (429/5xx) and tenacity `_resilient_get()` for application-level transient failure retries with exponential backoff. The tenacity decorator uses `reraise=True` so the original exception propagates to each method's try/except handler after retries exhaust. Tests mock `client.session.get` and the retry logic is exercised transparently.
- **Config constants**: All tunable numbers (PRR constants, API timeouts, retry config, drug-class/peer-discovery limits, Phase 1 drug identity API limits) are defined in `config.py` with named constants and clinical rationale comments. Modules import them — do not hardcode numeric values in FDA/PRR modules.
- **OTC Monograph Drugs Have No `drugsfda.json` Entry**: When resolving a drug identity, OTC monograph drugs (e.g., aspirin, ibuprofen) do not appear in `drug/drugsfda.json` because they are approved via a monograph pathway, not an NDA/ANDA/BLA. The resolution will succeed (finding brand/generic names) but `market_authorization_date` will be `None`, and `resolution_confidence` will be `PARTIAL`. This is correct behavior, not an error; document it when generating PSUR export.
- **Drug Identity Resolution Limitation**: When a drug name matches multiple NDAs/ANDAs/BLAs, the earliest `submission_status_date` across all matched applications with `submission_status="AP"` is used as the market_authorization_date. This can conflate different products with the same brand name (e.g., different dosage forms approved at different times). For canonical identity, prefer exact NDA/ANDA/BLA number over inferring from bulk name resolution.
- **Phase 2 PSUR Cycle Anchoring (International Birth Date Semantics)**: PSUR periods are anchored to the drug's market_authorization_date (its "International Birth Date" in ICH E2C(R2) PBRER terminology), NOT to calendar date or "today minus N months." The `compute_psur_period()` function selects the most recently *completed* fixed-length cycle as of today (e.g., if KEYTRUDA was approved on 2014-09-04 and today is 2024-07-13, the most recent completed 6-month period is 2024-03-04 to 2024-09-03). This matches real-world PSUR reporting practice — a critical distinction worth remembering for Phase 6 integration and preventing incorrect period selection in future enhancements.
- **Phase 3 Deduplication: Full-Set-Equality Heuristic Matching** — The `deduplicate_reports()` function's fallback heuristic pass (after exact-ID matching) treats two reports with *different* `report_id`s as duplicates only if their full normalized `drug_names` set AND full normalized `symptom_list` set AND `date` are ALL identical (full-set equality, not "any shared term"). This deliberately strict approach prevents over-merging genuinely distinct multi-symptom cases (e.g., a single patient's report with both headache and chest pain should not merge with a different patient's report mentioning only headache, even though they share one symptom). Known limitation: `date` here is `receivedate` (FDA receipt date), not a per-reaction adverse-event onset date — FAERS lacks a single unambiguous per-reaction event date, so this is the proxy used here and elsewhere in the codebase. Reports with empty `drug_names` or empty `symptom_list` are entirely excluded from heuristic matching (treated as always-unique).
- **Phase 2–3 `_merge_chunks()` vs. Phase 3 `deduplicate_reports()`** — The `_merge_chunks()` function (Phase 2) performs a version-aware exact-match merge on `report_id` across chunks *only* to handle boundary-day collisions (chunk date ranges are inclusive on both ends, so a report with event_date exactly on a boundary appears in two adjacent chunks). This is retrieval hygiene only — NOT a general deduplication feature. Do not confuse it with Phase 3's `deduplicate_reports()`, which adds content-level heuristic matching (normalized drug names + symptoms + date) and audit statistics. `_merge_chunks()` is scoped strictly to inter-chunk collision resolution and must never be extended for Phase 3+ dedup tasks.
- **Phase 4 Label Status Classification: Batch Processing for Ranking** — Phase 4 added `classify_label_statuses()` as a batch counterpart to the existing single-symptom `classify_label_status()`. The batch function reuses the substring-match logic rather than duplicating it, normalizes all input symptoms to uppercase (`.strip().upper()`), dedupes them case-insensitively, and sorts the unique set for deterministic iteration order (critical for reproducible signal rankings). A frozen dataclass `LabelClassificationResult` is returned with per-symptom status mapping and aggregate counts (labeled/unlabeled/unknown); the `total_symptoms == labeled_count + unlabeled_count + unknown_count` invariant is guaranteed structurally. The frozen dataclass design prevents accidental mutation by downstream Phase 5+ ranking logic. Symptom list input is never mutated (defensive); entries that are empty, whitespace-only, or non-string are filtered out before processing.
- **Phase 5 Deterministic Signal Ranking: Lexicographic Tier Ordering** — Phase 5's `ranking.py` uses a four-tier lexicographic sort (Seriousness & Outcome, Strength of Evidence, Reversibility, Public Health Impact) to produce a globally-ranked signal list without computing a single composite score. The sort key is a tuple of tier ordinals (e.g., `(seriousness_rank, evidence_rank, reversibility_rank, health_impact_rank)` where each element is an int from an ordered enum); this internal sort key enables deterministic ranking but is **never** stored on or exposed via `RankedSignal` or `RankingResult` dataclasses. Each tier is exported as a human-readable string field (e.g., `seriousness_tier="DEATH"`) for auditability and reverse-engineering the ranking. This design explicitly honors the scope doc's ban on a single composite "AdverseScore" risk number while providing deterministic, reproducible, audit-trail-friendly ranking. When new ranking criteria are added in future phases, extend the tuple lexicographically (append new tier ordinals to the right) — do not collapse multiple tiers into a weighted formula.

## Test Suite

Tests are organized under `tests/` directory, split by module (mirroring `src/adverse_score/` structure):

```
tests/
├── conftest.py                   # Pytest fixtures (all tests)
├── unit/
│   ├── test_fda_client.py        # FDAClient methods, query building, retry logic, PSUR retrieval (Phase 2), flatten/merge phase 3 fields, reactions/outcome extraction (Phase 5)
│   ├── test_deduplication.py     # deduplication.py exact-ID and heuristic dedup, audit trail (Phase 3)
│   ├── test_ranking.py           # ranking.py seriousness/evidence/reversibility/health-impact tiering, tie-breaks, edge cases (Phase 5)
│   ├── test_prr.py               # PRR + Wald 95% CI calculation
│   ├── test_label_classifier.py  # Label classification (single + batch, Phase 4)
│   ├── test_persistence.py       # AnalysisStore (currently skeleton)
│   ├── test_orchestrator.py      # orchestrator.py placeholder validation
│   ├── test_agent_tools.py       # agent_tools.py placeholder validation
│   └── test_drug_identity.py     # drug_identity.py resolution, error handling (Phase 1)
└── e2e/
    ├── test_fda_client_e2e.py     # Live openFDA API contract validation
    ├── test_fda_client_psur_e2e.py # PSUR chunked retrieval against live API (Phase 2)
    ├── test_deduplication_e2e.py  # Deduplication against live PSUR retrieval results (NEW, Phase 3)
    ├── test_prr_e2e.py            # PRR against live FAERS data
    └── test_drug_identity_e2e.py  # Drug identity resolution against live API (Phase 1)
```

**Current counts:** 138 unit tests (all passing, no API keys, ~24s), 16 E2E tests (all passing against live openFDA API, including Phase 2 PSUR retrieval and Phase 3 deduplication tests).

Configuration: `pytest.ini` specifies `pythonpath = src tests` (space-separated, not comma-separated — comma breaks pytest) and `testpaths = tests`.

```bash
# Unit tests only (fast, no API keys)
pytest tests/unit -v

# E2E integration tests (requires .env)
pytest tests/e2e -v -m e2e

# Full suite (from repo root)
pytest -v
```

Old root-level `test_adversescore.py`, `test_e2e.py`, `conftest.py` are deleted; content redistributed into new structure.
