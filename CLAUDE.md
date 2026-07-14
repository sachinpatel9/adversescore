# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdverseScore is being rebuilt as a **PSUR consolidation agent** for pharmacovigilance teams. Given a drug name and reporting period, it consolidates FAERS adverse event data, deduplicates case reports, ranks signals deterministically, and generates structured `.docx` documents aligned to ICH E2C(R2) PBRER format. The rebuild is tracked in the authoritative scope document `docs/PSUR_CONSOLIDATION_SCOPE.md` (11-phase plan). **Currently completed:** Phase 0 (Foundation Cleanup), Phase 1 (Drug Identity Resolution), and Phase 2 (Chunked, Paginated FDA Retrieval). Phases 3–11 (deduplication, ranking engine, document generation, new UI, agent orchestration) are in progress.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run app.py
```

Requires a `.env` file with `OPENFDA_API_KEY` and `OPENAI_API_KEY` (see `.env.example`).

## Architecture

Current state (Phase 0–2 complete): Foundation cleanup, drug identity resolution, and chunked FDA retrieval are done. Phase 3+ modules (deduplication, ranking, document generation, new agent orchestration) are planned but not yet built.

- **`app.py`** — Streamlit UI. Currently a placeholder chat interface (shows "PSUR consolidation agent is being rebuilt" message instead of invoking a live agent). Will be rebuilt in Phases 8–10 to display PSUR period selector, ranked signal list, and `.docx` export controls.
- **`src/adverse_score/drug_identity.py`** (NEW, Phase 1) — Given a raw drug name, resolves it to canonical brand/generic name variants and market authorization/approval date. Entry point: `resolve_drug_identity(raw_name: str, client: Optional[FDAClient] = None)`. Uses openFDA label/NDC queries (exact then broadened) followed by `drugsfda.json` for approval date lookup. Returns `DrugIdentity` dataclass (with resolution_confidence) or `DrugIdentityError` on failure. Known limitation: OTC monograph drugs (aspirin, ibuprofen) have no `drugsfda.json` entry, so approval_date is `None` (marked `PARTIAL` confidence, not an error).
- **`src/adverse_score/client.py`** — Much-reduced orchestrator. Removed: `calculate_final_score`, `get_peer_benchmark`, `fetch_quarterly_data`, `compute_trend`, all scoring-related methods. Retained: FDA delegation one-liners (`fetch_events`, `fetch_label_text`, `build_query`, etc.), `_classify_label_status`, `_calculate_prr_metrics`. No longer requires API keys at construction time (backward-compatible skeleton pattern).
- **`src/adverse_score/fda_client.py`** — openFDA HTTP client (~500 lines). Existing methods: `fetch_events`, `fetch_label_text`, `_discover_drug_class`, `_discover_peers`, `_fetch_symptom_counts`, `build_query`, `_flatten_results`, `_compute_quarter_boundaries`. Dual-layer retry (urllib3 for transport codes, tenacity for app-level transience). Phase 2 additions: module-level pure functions `_add_months()` (stdlib day-of-month clamping), `compute_psur_period()` (anchors PSUR cycle to market_authorization_date, selecting most recently completed fixed-length cycle; falls back to rolling lookback for OTC or pre-approval drugs), `_compute_psur_chunks()` (slices PSUR period into quarterly chunks with labels `"chunk-{i}-YYYYMMDD-to-YYYYMMDD"`), `_merge_chunks()` (first-seen-wins dedup on report_id for boundary collisions). New `FDAClient` methods: `_build_psur_chunk_query()` (composes OR'd drug name variants + AND'd date range via params dict), `_fetch_chunk_paginated()` (paginates single chunk with skip/limit capped at `PSUR_SKIP_CEILING=25000`; aborts on per-chunk error without failing other chunks). New public method: `fetch_psur_reports()` (top-level entry point orchestrating all of the above; returns `PSURRetrievalResult` never raises except ValueError for empty name_variants list). Two new frozen dataclasses: `ChunkResult` (per-chunk metadata: label, dates, reports, counts, truncated flag, optional error) and `PSURRetrievalResult` (aggregate: period bounds, all chunks, merged reports, totals, completeness flags). Layering note: does NOT import `drug_identity.py` (avoids circular import); accepts plain primitives (name variants list, optional date, period string) instead.
- **`src/adverse_score/prr.py`** — Pure PRR + Wald 95% CI math (~70 lines). `calculate_prr(drug_counts, class_counts, target_symptom, label_text)`. Unchanged; will feed Phase 3+ ranking engine.
- **`src/adverse_score/label_classifier.py`** — Pure label classification (~30 lines). Now contains only `classify_label_status` (LABELED/UNLABELED/LABEL_STATUS_UNKNOWN). Removed: `calculate_label_penalty` (was old scoring logic, not needed for new ranking).
- **`src/adverse_score/orchestrator.py`** — Placeholder. `agent_executor = None` sentinel; importing no longer requires API keys. Will be rebuilt in Phase 8 with new LangGraph agent, multi-turn guidance, and 7-guardrail system prompt.
- **`src/adverse_score/persistence.py`** — Reduced to bare skeleton: `__init__`, no-op `_init_schema`, context-manager protocol. Old `AnalysisStore.analyses` table and CRUD methods removed. Phase 7 will design new schema for consolidated datasets + conversation history.
- **`src/adverse_score/logger.py`** — JSON-structured logging to stderr. Unchanged.
- **`src/adverse_score/config.py`** — Removed 26 old composite-score constants (severity weights, label penalties, recency decay, confidence curve, guardrail thresholds, etc.). Retained: `initialize_config()`, PRR constants, API timeout/retry constants, drug-class/peer-discovery constants. Added Phase 1 constants: `DRUGSFDA_ENDPOINT`, `NDC_ENDPOINT`, `DRUG_IDENTITY_*_LIMIT`, `DRUGSFDA_APPROVED_STATUS`. Added Phase 2 constants (PSUR Chunked Retrieval): `PSUR_PAGE_SIZE=1000`, `PSUR_SKIP_CEILING=25000`, `PSUR_CHUNK_MONTHS=3`, `PSUR_PERIOD_MONTHS` (dict: period string → month counts), `PSUR_PERIOD_FALLBACK_DAYS` (dict: period string → fallback day counts, used only on anchor-less path).

## Workflow Multi-Agent Orchestration

Each phase's implementation plan must declare a "Critical Files" list (files that may be
created/modified) before any subagent is dispatched. All three subagents below operate
strictly within that declared file boundary unless a subagent discovers a genuine
cross-cutting bug — in which case it stops and reports back rather than silently expanding
scope.

1. Build SubAgent:
* Model: Sonnet 5 (high effort)
* Scope: Implements the code + tests described in the phase's plan, touching only the files
  in that phase's "Critical Files" list.
* Out of scope: documentation files (CLAUDE.md, README.md), modules outside the declared
  file list, and already-passing tests unrelated to the phase's own changes.
* Produces: working code + a passing test suite for the declared scope, plus a concise
  summary of what was built (function/class names, file paths) for the Review SubAgent.

2. Review SubAgent:
* Model: Fable 5 OR Opus 4.8 (medium effort) (use Opus 4.8 if usage runs out for Fable 5
  during the build)
* Scope: Reviews only the diff produced by the Build SubAgent, against the originating
  phase plan's acceptance criteria and test plan — not a general-purpose codebase audit.
  Acts as a skeptical Staff Developer: checks correctness, security invariants
  (`_sanitize_for_query()`, `params=` transport conventions), layering invariants, and test
  coverage against the plan's stated test cases. Uses the Staff-Python-Review skill.
* Fixes bugs it finds directly, within the same file boundary the Build SubAgent used —
  does not expand into unrelated refactors.
* Out of scope: documentation files. Runs only after the Build SubAgent reports completion.
* Produces: a short report of what was found/fixed, handed to the Write SubAgent.

3. Write SubAgent:
* Model: Haiku (Medium Effort)
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
starting. ALL tools are available for each subagent to use for any purposes necessary.




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
- **Phase 2 `_merge_chunks()` Dedup is Not Phase 3 Dedup**: The `_merge_chunks()` function performs a first-seen-wins exact-match merge on `report_id` across chunks only to handle boundary-day collisions (chunk date ranges are inclusive on both ends, so a report with event_date exactly on a boundary appears in two adjacent chunks). This is NOT a general deduplication feature and must not be confused with Phase 3's planned deduplication engine (which will handle `safetyreportid`, heuristic fallback matching on normalized name + MedDRA + date, and full content-level near-duplicate detection with audit statistics). Do not extend or repurpose `_merge_chunks()` for dedup tasks — it is scoped only to retrieval hygiene.

## Test Suite

Tests are organized under `tests/` directory, split by module (mirroring `src/adverse_score/` structure):

```
tests/
├── conftest.py                   # Pytest fixtures (all tests)
├── unit/
│   ├── test_fda_client.py        # FDAClient methods, query building, retry logic, PSUR retrieval (Phase 2)
│   ├── test_prr.py               # PRR + Wald 95% CI calculation
│   ├── test_label_classifier.py  # Label classification
│   ├── test_persistence.py       # AnalysisStore (currently skeleton)
│   ├── test_orchestrator.py      # orchestrator.py placeholder validation
│   ├── test_agent_tools.py       # agent_tools.py placeholder validation
│   └── test_drug_identity.py     # drug_identity.py resolution, error handling (Phase 1)
└── e2e/
    ├── test_fda_client_e2e.py     # Live openFDA API contract validation
    ├── test_fda_client_psur_e2e.py # PSUR chunked retrieval against live API (NEW, Phase 2)
    ├── test_prr_e2e.py            # PRR against live FAERS data
    └── test_drug_identity_e2e.py  # Drug identity resolution against live API (Phase 1)
```

**Current counts:** 81 unit tests (all passing, no API keys, ~24s), 15 E2E tests (all passing against live openFDA API, including 2 new PSUR retrieval tests covering anchored-period and OTC-fallback paths).

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
