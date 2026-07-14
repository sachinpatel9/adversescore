# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdverseScore is being rebuilt as a **PSUR consolidation agent** for pharmacovigilance teams. Given a drug name and reporting period, it consolidates FAERS adverse event data, deduplicates case reports, ranks signals deterministically, and generates structured `.docx` documents aligned to ICH E2C(R2) PBRER format. The rebuild is tracked in the authoritative scope document `docs/PSUR_CONSOLIDATION_SCOPE.md` (11-phase plan). **Currently completed:** Phase 0 (Foundation Cleanup), Phase 1 (Drug Identity Resolution), Phase 2 (Chunked, Paginated FDA Retrieval), Phase 3 (Deduplication), Phase 4 (Label Status Classification), Phase 5 (Deterministic Ranking Engine), Phase 6 (Consolidation Orchestration), Phase 7 (Persistence Layer Redesign), Phase 8 (Agent Orchestration & Guardrails), and Phase 9 (Document Generation). Phases 10–11 (new UI, full validation) are in progress.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run app.py
```

Requires a `.env` file with `OPENFDA_API_KEY` and `OPENAI_API_KEY` (see `.env.example`).

## Architecture

Current state (Phase 0–9 complete): Foundation cleanup, drug identity resolution, chunked FDA retrieval, deduplication, label status classification, deterministic ranking engine, consolidation orchestration, persistence layer, agent orchestration with guardrails, and document generation are done. Phases 10–11 modules (UI rebuild, full validation) are planned but not yet built.

- **`app.py`** — Streamlit UI. Currently a placeholder chat interface (shows "PSUR consolidation agent is being rebuilt" message instead of invoking a live agent). Will be rebuilt in Phases 10–11 to display PSUR period selector, ranked signal list, and `.docx` export controls.
- **`src/adverse_score/drug_identity.py`** (NEW, Phase 1) — Given a raw drug name, resolves it to canonical brand/generic name variants and market authorization/approval date. Entry point: `resolve_drug_identity(raw_name: str, client: Optional[FDAClient] = None)`. Uses openFDA label/NDC queries (exact then broadened) followed by `drugsfda.json` for approval date lookup. Returns `DrugIdentity` dataclass (with resolution_confidence) or `DrugIdentityError` on failure. Known limitation: OTC monograph drugs (aspirin, ibuprofen) have no `drugsfda.json` entry, so approval_date is `None` (marked `PARTIAL` confidence, not an error).
- **`src/adverse_score/client.py`** — Much-reduced orchestrator. Removed: `calculate_final_score`, `get_peer_benchmark`, `fetch_quarterly_data`, `compute_trend`, all scoring-related methods. Retained: FDA delegation one-liners (`fetch_events`, `fetch_label_text`, `build_query`, etc.), `_classify_label_status`, `_calculate_prr_metrics`. No longer requires API keys at construction time (backward-compatible skeleton pattern).
- **`src/adverse_score/fda_client.py`** — openFDA HTTP client (~500 lines). Existing methods: `fetch_events`, `fetch_label_text`, `_discover_drug_class`, `_discover_peers`, `_fetch_symptom_counts`, `build_query`, `_flatten_results`, `_compute_quarter_boundaries`. Dual-layer retry (urllib3 for transport codes, tenacity for app-level transience). Phase 2 additions: module-level pure functions `_add_months()` (stdlib day-of-month clamping), `compute_psur_period()` (anchors PSUR cycle to market_authorization_date, selecting most recently completed fixed-length cycle; falls back to rolling lookback for OTC or pre-approval drugs), `_compute_psur_chunks()` (slices PSUR period into quarterly chunks with labels `"chunk-{i}-YYYYMMDD-to-YYYYMMDD"`), `_merge_chunks()` (first-seen-wins dedup on report_id for boundary collisions). New `FDAClient` methods: `_build_psur_chunk_query()` (composes OR'd drug name variants + AND'd date range via params dict), `_fetch_chunk_paginated()` (paginates single chunk with skip/limit capped at `PSUR_SKIP_CEILING=25000`; aborts on per-chunk error without failing other chunks). New public method: `fetch_psur_reports()` (top-level entry point orchestrating all of the above; returns `PSURRetrievalResult` never raises except ValueError for empty name_variants list). Two new frozen dataclasses: `ChunkResult` (per-chunk metadata: label, dates, reports, counts, truncated flag, optional error) and `PSURRetrievalResult` (aggregate: period bounds, all chunks, merged reports, totals, completeness flags). Layering note: does NOT import `drug_identity.py` (avoids circular import); accepts plain primitives (name variants list, optional date, period string) instead. Phase 3 additions: `_flatten_results()` now emits three new keys on each flattened report dict (`symptom_list` as a list of MedDRA PT terms, `drug_names` as a deduped list of medicinal product names, `safetyreportversion` as int), and `_merge_chunks()` is now version-aware (when same `report_id` recurs across chunks, higher `safetyreportversion` wins instead of blind first-seen-wins). Two new module-level helpers: `_extract_drug_names(report)`, `_parse_version(raw)`. Phase 5 additions: `_flatten_results()` now also extracts per-report `reactions` key (`[{"term": <MedDRA PT>, "outcome_code": <int 1-6 or None>}, ...]`) from raw FAERS `patient.reaction[].reactionoutcome` field via new `_parse_outcome_code(raw)` helper (never raises; returns `None` for missing/malformed/out-of-range values, enabling safe reversibility tiering for Phase 5 ranking).
- **`src/adverse_score/deduplication.py`** (NEW, Phase 3) — Pure deduplication module (no HTTP, no `FDAClient` import), same architectural pattern as `prr.py`/`label_classifier.py`. Removes duplicate FAERS case reports from a flattened report list. Entry point: `deduplicate_reports(reports: list) -> DedupResult`. Two-pass implementation: (1) Exact `safetyreportid` match with version awareness (groups by `report_id`, keeps highest `safetyreportversion` per id; reports with missing/`None` `report_id` get unique synthetic keys and never collapse), (2) Fallback heuristic match on survivors only (two reports with different `report_ids` are duplicates only if their full normalized `drug_names` set AND full normalized `symptom_list` set AND `date` are ALL identical — deliberately strict to avoid over-merging). Reports with empty `drug_names` or empty `symptom_list` are excluded from heuristic matching. Returns `DedupResult` (frozen dataclass): `reports`, `total_input_count`, `total_output_count`, `removed_by_exact_id`, `removed_by_heuristic`, `audit_trail` (list of removal dicts). Known limitation: `date` here is `receivedate` (FDA receipt date), not a true adverse-event onset date—this is the same proxy used elsewhere in the codebase.
- **`src/adverse_score/ranking.py`** (NEW, Phase 5) — Deterministic signal ranking engine (no HTTP, no `FDAClient` import), pure module importing only `dataclasses`, `.config`, `.prr`, `.label_classifier`, `.logger`. Entry point: `rank_signals(reports: list, class_counts: dict, label_text: str) -> RankingResult`. A "signal" is one unique MedDRA PT symptom across deduplicated reports. Ranks each signal across four lexicographically-ordered tiers: (1) **Seriousness & Outcome** — DEATH > HOSPITALIZATION > OTHER_SERIOUS > NON_SERIOUS (derived from existing `is_death`/`is_hospitalization`/`severity` fields), (2) **Strength of Evidence** — STRONG_UNLABELED > STRONG_LABELED > WEAK_UNLABELED > WEAK_LABELED > UNKNOWN_LABEL_STATUS (reuses `prr.py`'s `calculate_prr()` output; unlabeled outranks labeled at equivalent evidence strength per scope doc), (3) **Reversibility** — FATAL > POOR > REVERSIBLE > UNKNOWN (derived from newly-extracted per-reaction FAERS outcome codes via Phase 5 `fda_client.py` changes; UNKNOWN used when outcome data is absent, never silently assumed reversible), (4) **Public Health Impact** — HIGH > MODERATE > LOW (derived from report-volume thresholds: 100/20). Returns `RankedSignal` (frozen dataclass: `symptom`, `rank`, `seriousness_tier`, `strength_of_evidence_tier`, `reversibility_tier`, `public_health_tier`, `prr_metrics` dict, `report_count`) and `RankingResult` (frozen dataclass: `ranked_signals` list, `label_summary` as `LabelClassificationResult`, `formula_version` string, `total_signals` int). Sort key is purely internal (tuple of tier ordinals); never stored/exposed on output dataclasses — this design explicitly avoids the banned single composite score. Known limitation: reversibility is a heuristic approximation based on outcome codes; true clinical reversibility assessment requires expert review.
- **`src/adverse_score/consolidation.py`** (NEW, Phase 6) — Single pipeline entry point wiring Phases 1–5 together into one coherent, callable PSUR consolidation workflow. Entry point: `consolidate_psur(drug_name: str, period: str, client: Optional[FDAClient] = None) -> Union[ConsolidationResult, ConsolidationError]`. Accepts optional `FDAClient` (dependency-injection pattern matching `drug_identity.py`); if none provided, constructs one (catching `EnvironmentError` → structured error). Pipeline: `resolve_drug_identity()` (Phase 1) → builds deduped `name_variants` from brand/generic/substance names → `fetch_psur_reports()` (Phase 2) → `deduplicate_reports()` (Phase 3) → `fetch_label_text()` + `_discover_drug_class()` + conditional `_fetch_symptom_counts()` (canonical_name only, no fallback loop; known limitation: generic-only drugs with no brand_names may get empty label lookup since `fetch_label_text` queries brand_name field specifically; degrades gracefully to LABEL_STATUS_UNKNOWN downstream, never crashes) → `rank_signals()` (Phase 5). Returns `ConsolidationResult` (frozen dataclass: `drug_identity`, `period`, `retrieval`, `dedup`, `ranking`, `pharm_class`, `class_counts_available` computed as `bool(class_counts)` not `bool(pharm_class)`, `formula_version` passthrough) or `ConsolidationError` (frozen dataclass: `drug_name`, `stage`, `reason`, `message`). Follows established "structured result, never raise for domain failures" convention. Empty report set after deduplication is NOT an error — produces valid `ConsolidationResult` with `ranking.total_signals == 0`. No new `config.py` constants added (pure wiring, no new tunable thresholds).
- **`src/adverse_score/prr.py`** — Pure PRR + Wald 95% CI math (~70 lines). `calculate_prr(drug_counts, class_counts, target_symptom, label_text)`. Unchanged; will feed Phase 4+ ranking engine.
- **`src/adverse_score/label_classifier.py`** — Pure label classification. Contains `classify_label_status(label_text: str, symptoms_str: str) -> str` (single symptom, LABELED/UNLABELED/LABEL_STATUS_UNKNOWN via substring match; unchanged from prior phases, used by prr.py) and NEW Phase 4: `classify_label_statuses(label_text: str, symptoms: list) -> LabelClassificationResult` (batch variant for Phase 5 ranking consumption). The batch function normalizes symptoms via `.strip().upper()`, dedupes case-insensitively, filters empty entries, sorts for determinism, and calls the existing single-symptom function once per unique symptom. Returns frozen dataclass `LabelClassificationResult` with per-symptom status dict, total/labeled/unlabeled/unknown counts. Invariant: `labeled_count + unlabeled_count + unknown_count == total_symptoms`.
- **`src/adverse_score/orchestrator.py`** (REBUILT, Phase 8) — LangGraph-based conversational agent layer enforcing seven guardrails from the scope doc. Entry point: `run_agent_turn(conversation_history: list, user_message: str, in_session_dataset: Optional[ConsolidationResult] = None) -> AgentTurnResult` — a PURE function with zero dependency on Phase 7's persistence layer. Persistence (saving messages, consolidations) is entirely the caller's responsibility, matching this codebase's dependency-injection pattern. `AgentTurnResult` (frozen dataclass): `response_text`, `updated_dataset` (current in-session consolidation after this turn), `dataset_changed` (True if new consolidation happened, signals caller to persist via `ConsolidationStore.save_consolidation()`). Two agent tools provided by `agent_tools.py`: `resolve_drug_identity_tool()` and `consolidate_psur_tool()` (both return JSON-serializable dicts with bounded summaries — top 20 ranked signals for PSUR tool to avoid LLM context window bloat). Novel technical design: contextvars.ContextVar capture box (installed per-call, mutated by tool nodes) carries full `ConsolidationResult` across LangGraph's tool-execution boundary; threading.local() was tried but failed because tool nodes run on different worker thread. Guardrail enforcement: Guardrails 1 (Causality Lock) and 7 (Scope Enforcement) are system-prompt-enforced with log-only defense-in-depth (`_flag_causal_language()`); Guardrails 4/5/6 (Human Review, Completeness, Formula Version Audit Trail) are code-enforced unconditionally at turn-end via deterministic disclosure functions (sourcing real dataclass fields), guaranteeing these three guardrails always present structurally independent of LLM memory; Guardrails 2/3 (No-Fabrication Placeholder, Draft Marking) are document-level (Phase 9). Installed LangGraph 1.1.0, LangChain 1.2.11, langchain-openai 1.1.11. Config additions: `OPENAI_CHAT_MODEL="gpt-4o"`, `AGENT_TEMPERATURE=0.1`, `AGENT_MAX_ITERATIONS=5` (runaway tool-calling terminates at `AGENT_MAX_ITERATIONS + 1` calls per LangGraph 1.1.0 behavior). Known limitation: Guardrail 1 (causality) is fundamentally prompt-enforced, not code-guaranteed — log-only flagging provides audit trail but never blocks response, so "no causality" relies on GPT-4o following system prompt. A real bug was found and fixed during development: completeness/formula-version disclosure suppressed if LLM response contained word "retrieved" in passing (e.g. "5 signals were retrieved and ranked...") without stating real completeness data — fixed by requiring evidence of BOTH completeness AND formula-version markers before skipping disclosure append; regression tests added. Import-without-API-keys safety: `ChatOpenAI` client constructed lazily on first call, not at import time.
- **`src/adverse_score/agent_tools.py`** (NEW, Phase 8) — Two LangGraph-compatible tools wrapping Phase 1–7 pipeline modules directly, reimplementing no logic. `resolve_drug_identity_tool(drug_name: str)` wraps Phase 1's `resolve_drug_identity()` for lightweight identity questions without running full consolidation. `consolidate_psur_tool(drug_name: str, period: str)` wraps Phase 6's `consolidate_psur()` (primary data-gathering tool). Both tools return JSON-serializable plain dicts (dates converted to ISO strings; `ConsolidationError`/`DrugIdentityError` surfaced as graceful `{"error": ...}` dicts, never crashes). `consolidate_psur_tool` returns a bounded summary (top 20 ranked signals, completeness counts, dedup stats) rather than full nested dataset to avoid LLM context window bloat; the full `ConsolidationResult` is captured separately via contextvars mutation (see orchestrator.py). Exports `begin_consolidation_capture()` function (called at start of every `run_agent_turn()`) to install fresh per-call capture box.
- **`src/adverse_score/document_generator.py`** (NEW, Phase 9) — Generates `.docx` draft PSUR documents aligned to the verified ICH E2C(R2) PBRER structure. Entry point: `generate_psur_document(result: ConsolidationResult) -> Document` (returns a `python-docx` `Document` object). Trimmed design: real data populates Section 1 (Introduction), Section 6 (Data in Summary Tabulations), Section 15 (Overview of Signals: real data table of full ranked signals, uncapped), and Section 16.1-16.3 (Signal and Risk Evaluation: LLM-narrated with evidentiary framing). All other sections (2–5, 7–14, 16.4–19, 20/Appendices) use one consolidated placeholder note (Guardrail 2, exact literal text). Draft marking embedded twice: structurally in docx header/footer AND in cover-page body paragraph (Guardrail 3). Completeness data, deduplication methodology, ranking formula version, and mandatory human-review disclaimer are all deterministically appended (Guardrails 4, 5, 6). LLM narration via single `_narrate_signals()` call (not a full agent) with new `DOCUMENT_NARRATION_TEMPERATURE = 0.0` constant and fabrication guardrails (`_check_narration_for_fabricated_entities` scans against allowed vocabulary, reimplemented causal-language check). New config constants: `DOCUMENT_NARRATION_TEMPERATURE`, `PBRER_PLACEHOLDER_TEXT`, `PBRER_DRAFT_MARKING_TEXT`, `PBRER_OMITTED_SECTIONS_NOTE`. New dependency: `python-docx` 1.2.0. Tests: 34 unit tests covering all PBRER sections present, placeholder byte-exact (never LLM-touched), draft marking structural tests, completeness/formula version from real fixtures, full ranked-signal table uncapped, OTC/None handling, empty signals, and guardrail functions tested directly. E2E: 1 live test with KEYTRUDA/6mo producing 2,548-row signal table in 84KB `.docx`.
- **`src/adverse_score/persistence.py`** (REBUILT, Phase 7) — Persists consolidated PSUR results and conversation history to SQLite for cross-session memory, enabling multi-turn follow-ups without re-querying FDA. `AnalysisStore` renamed to `ConsolidationStore`. Two-table schema: (1) `consolidations` table (`id` autoincrement PK, `canonical_name`, `period`, `created_at` ISO 8601, `result_json`), indexed on `(canonical_name, period, created_at DESC)`; (2) `conversation_messages` table (`id` autoincrement PK, `session_id`, `consolidation_id` nullable FK, `role`, `content`, `created_at`), indexed on `(session_id, id)`, ordered by insertion order (never `created_at`) to avoid same-timestamp collisions. Serializes `ConsolidationResult` (frozen dataclass tree of 7 nested types: `DrugIdentity`, `PSURRetrievalResult`, `ChunkResult`, `DedupResult`, `RankingResult`, `RankedSignal`, `LabelClassificationResult`) via `dataclasses.asdict()` with two manual post-processing steps: (1) `DrugIdentity.market_authorization_date` (a `date` object) → ISO string on save, parsed back via `date.fromisoformat()` on load; `None` (OTC drugs) passes through untouched in both directions. (2) `PSURRetrievalResult.chunks[].reports` (pre-merge per-chunk report lists) trimmed to `[]` on save — this data is a strict subset already folded into merged `retrieval.reports` and deduplicated `dedup.reports`, so omitting it is a **documented, deliberate exception to literal round-trip fidelity**, avoiding ~2–3x storage bloat for high-volume drugs. Deserialization reconstructs full `ConsolidationResult` objects (not plain dicts) via bottom-up nested dataclass reconstruction, so loaded consolidations are indistinguishable from freshly-returned ones. Public methods: `save_consolidation(result: ConsolidationResult) -> int` (inserts new row every time, never upserts/overwrites; full history retained), `load_consolidation(consolidation_id: int) -> Optional[ConsolidationResult]`, `get_latest_consolidation(canonical_name: str, period: str) -> Optional[ConsolidationResult]`, `list_consolidations(canonical_name=None) -> list` (lightweight metadata only, no JSON decode), `save_message(session_id, role, content, consolidation_id=None) -> int`, `get_conversation_history(session_id: str) -> list`. `ConsolidationError` never persisted — only successful `ConsolidationResult` objects saved. Tests: 18 tests covering schema creation, full round-trip fidelity on real nested instances, `None` market-authorization-date round-trip, empty ranked-signals round-trip, multiple saves never overwriting, `get_latest_consolidation` correctness, `list_consolidations` filtering, and interleaved-session conversation ordering.
- **`src/adverse_score/logger.py`** — JSON-structured logging to stderr. Unchanged.
- **`src/adverse_score/config.py`** — Removed 26 old composite-score constants (severity weights, label penalties, recency decay, confidence curve, guardrail thresholds, etc.). Retained: `initialize_config()`, PRR constants, API timeout/retry constants, drug-class/peer-discovery constants. Added Phase 1 constants: `DRUGSFDA_ENDPOINT`, `NDC_ENDPOINT`, `DRUG_IDENTITY_*_LIMIT`, `DRUGSFDA_APPROVED_STATUS`. Added Phase 2 constants (PSUR Chunked Retrieval): `PSUR_PAGE_SIZE=1000`, `PSUR_SKIP_CEILING=25000`, `PSUR_CHUNK_MONTHS=3`, `PSUR_PERIOD_MONTHS` (dict: period string → month counts), `PSUR_PERIOD_FALLBACK_DAYS` (dict: period string → fallback day counts, used only on anchor-less path). Added Phase 5 constants (Deterministic Ranking Engine): `RANKING_FORMULA_VERSION` (current: "1.0"), seriousness/strength-of-evidence/reversibility/public-health tier name constants plus their fixed priority-order tuples, `REACTION_OUTCOME_*` FAERS outcome code constants (1-6), public-health volume thresholds (100 for HIGH, 20 for MODERATE, remaining as LOW).

## Workflow Multi-Agent Orchestration

Each phase's implementation plan must declare a "Critical Files" list (files that may be
created/modified) before any subagent is dispatched. All subagents below operate strictly
within that declared file boundary unless a subagent discovers a genuine cross-cutting bug
— in which case it stops and reports back (with file path, line number, and rationale)
rather than silently expanding scope.

**Dispatch mechanics:** Subagents are dispatched via the `Agent` tool with `subagent_type:
general-purpose` (or `claude`). Never use `Explore` or `Plan` types — those lack Write/Edit
tools. **Dispatch context optimization:** each subagent receives only phase-scoped context
(phase plan + Critical Files + relevant design decisions), not full CLAUDE.md, to minimize
token waste on repeated architecture overview.

**Handoff protocol:** Each subagent produces a **structured summary** for the next subagent,
following the templates below. This prevents re-stating facts and enforces clarity.

---

### Build SubAgent

* **Model:** Sonnet 5 (high effort)
* **Tool access:** Read, Write, Edit, Bash — full access needed for code generation and testing
* **Scope:** Implements code + tests per the phase plan, touching ONLY files in the Critical
  Files list. No documentation files (CLAUDE.md, README.md). No pre-existing tests unrelated
  to the phase's changes.
* **Test execution:** Run tests ONLY for modified Critical Files:
  ```bash
  pytest tests/unit/test_<phase_module>.py tests/e2e/test_<phase_module>_e2e.py -v
  ```
  Do NOT run full test suite (`pytest -v`) — that introduces unrelated flakes and wastes tokens.
* **Boundary validation:** Before committing, run `git status | grep modified` and verify EVERY
  file is in the Critical Files list. If any file is outside the list, STOP and report the
  boundary violation to the orchestrator with file path and rationale.
* **Produces:** Structured handoff summary (see template below)

**Build → Review Handoff Template:**
```
## Critical Files Modified
- file_path: <path>, <lines_added>/<lines_modified>
- (repeat for each file)

## Test Results
- Unit tests: N passed
- E2E tests: M passed
- Coverage: <module> at X%, <module> at Y%

## Known Issues or Uncertainties
(none) OR (list any, with rationale for leaving unresolved)

## Notes for Reviewer
(optional: edge cases manually tested, complex logic sections, assumptions made)
```

---

### Review SubAgent

* **Model:** Claude Sonnet 5 (must be explicit; if unavailable use Claude Opus 4.8)
* **Tool access:** Read, Write, Edit, Bash — full access for deep analysis and direct code fixes
* **Scope:** Reviews ONLY the diff produced by Build, against the phase plan's acceptance
  criteria and test plan (not a general-purpose codebase audit). Checks:
  - **Correctness:** Do the core algorithms work as designed? (hand-trace complex logic)
  - **Security invariants:** `_sanitize_for_query()` on all FDA queries, `params=` dict usage,
    no API keys hardcoded
  - **Layering invariants:** Pure modules (dedup, ranking, label_classifier, prr) have no
    HTTP imports; edge cases documented (None values, empty collections, missing fields)
  - **Test coverage:** Do the tests cover the highest-risk acceptance criteria? (e.g.,
    dedup over-merge guard, version-tiebreak correctness)
* **Skill usage:** MUST invoke `staff-python-reviewer` skill (project-scoped version in
  `.claude/skills/staff-python-reviewer/SKILL.md`) — do not review without it.
* **Fixes:** Apply bugs found directly, within the same file boundary Build used. Do NOT
  expand into unrelated refactors or touch documentation files.
* **Runs:** Only after Build reports completion (via handoff summary).
* **Produces:** Structured handoff summary (see template below)

**Review → Auditor Handoff Template:**
```
## Review Findings
- Bugs found and fixed: N (list with file:line if any)
- Security/layering issues: (none) OR (list with severity)
- Test coverage gaps: (none) OR (describe shortfall vs. acceptance criteria)

## Acceptance Criteria Status
- All criteria testable? YES / NO (if NO, explain which criteria and why)
- Tests comprehensive? YES / NEEDS_CLARIFICATION / NO

## Ready for Auditor?
YES (proceed to acceptance audit) OR NO (specify issues to resolve)

## Notes
(optional: hand-traced logic, complex correctness reasoning, assumptions verified)
```

---

### Acceptance Criteria Auditor (Fable)

* **Model:** Fable 5
* **Tool access:** Read ONLY (no Write, no Edit, no Bash) — this is a verification role, not
  a modification role
* **Scope:** Given the phase plan's acceptance criteria and Review's test results, verify
  that each criterion is actually satisfied (not just that tests pass in isolation). Acts as
  a skeptical requirements auditor: asks "did we really meet this criterion, or did we pass
  the test on a technicality?"
* **Method:** For each acceptance criterion:
  1. Read the criterion from the phase plan
  2. Read the corresponding test(s) and their results
  3. Hand-trace the code path to verify it satisfies the criterion
  4. Report: criterion met (YES) or unclear/unmet (NEEDS_REVIEW)
  
  **Example:** Criterion "heuristic dedup never merges partial symptom overlaps" + test
  `test_partial_symptom_overlap_does_not_merge` PASSED + code review of heuristic matching
  → conclusion: "Criterion met via frozenset equality check in dedup.py line X"

* **Produces:** Structured handoff summary (see template below)

**Auditor → Write Handoff Template:**
```
## Acceptance Criteria Audit Results
(For each criterion in the phase plan:)
- Criterion: "<text>"
  Status: MET / UNMET / UNCLEAR
  Evidence: (test name and result, hand-traced code path, or issue)

## Overall Audit Result
ALL CRITERIA MET ✓ (proceed to Write)
OR
CRITERIA X, Y REQUIRE REVIEW (send back to Review for clarification/fixes)

## Edge Case Notes
(optional: which acceptance criteria were highest-risk? Any that were barely met?)
```

---

### Write SubAgent

* **Model:** Haiku 4.5 (medium effort)
* **Tool access:** Read, Edit ONLY — cannot Write to arbitrary files or run Bash. Permitted
  files: CLAUDE.md, README.md, `docs/TODO.md` ONLY.
* **Scope:** Updates CLAUDE.md, README.md, and `docs/TODO.md` using the Build/Review/Auditor
  summary as its factual source. Never re-reads codebase, never re-derives architecture,
  never makes inferences beyond what the handoff summary states. If ambiguity exists, ask for
  clarification (via sendMessage to orchestrator or return summary with [NEEDS_CLARIFICATION]
  tags) rather than guessing.
* **Updates required:** Review's handoff specifies which CLAUDE.md sections change (e.g.,
  "add deduplication.py bullet line 30", "update test counts to 99 unit, 16 E2E"). Write
  makes those edits. README.md updates follow the existing pattern (add phase section,
  mark roadmap item, update file tree, update test counts). `docs/TODO.md` updates flip the
  completed phase's status marker from 🚧 to ✅, check off its Build/Tests items, and mark
  the next phase 🚧 if immediately starting.
* **Documentation quality:** Thorough in content, concise in structure, professional tone.
  Follow the style of prior phases exactly.
* **Runs:** Only after Auditor confirms all acceptance criteria are met.
* **Produces:** Done (no handoff needed; docs are the output).

---

### Orchestration Flow (Phase Execution)

```
User approves phase plan (via ExitPlanMode)
           ↓
Dispatch Build SubAgent (Sonnet 5)
  - Context: phase plan + Critical Files list + security invariants
  - Task: implement code + tests for Critical Files only
  - Timeout: ~30-60 min (depends on phase scope)
           ↓
Build completes → Handoff Summary
           ↓
Dispatch Review SubAgent (Sonnet 5)
  - Context: Build summary + phase plan + acceptance criteria
  - Task: deep review + fixes + staff-python-reviewer skill
  - Prerequisite: MUST use skill (non-negotiable)
  - Timeout: ~20-40 min
           ↓
Review green (tests pass) → Auditor Handoff Summary
           ↓
Dispatch Acceptance Criteria Auditor (Fable 5)
  - Context: Auditor handoff + acceptance criteria + code paths
  - Task: verify each criterion is actually met (not just tested)
  - Read-only (no modifications)
  - Timeout: ~10-15 min
           ↓
Auditor clears all criteria → Write Handoff Summary
           ↓
Dispatch Write SubAgent (Haiku 4.5)
  - Context: Auditor summary + CLAUDE.md/README.md sections to update
  - Task: update docs per handoff specification
  - Timeout: ~5-10 min
           ↓
Write completes → Orchestrator spot-checks CLAUDE.md accuracy
           ↓
Commit ready (user decides: commit/push/iterate)
```

---

### Token Optimization & Design Rationale

**Why eliminate the Orchestrator Agent?** If the plan is pre-approved via ExitPlanMode,
re-planning wastes ~50-100K tokens. Direct dispatch from the main Claude agent is faster
and cheaper.

**Why context-scoped handoffs?** Sending full CLAUDE.md to every agent (~160 lines per call)
is ~40-50% token waste on repeated architecture overview. Each agent gets only what it
needs.

**Why restrict Build test scope?** Running `pytest -v` (all 99+ tests) every phase adds
3-5 min and risk of unrelated test flakes. Running only tests for modified files keeps
feedback tight and deterministic.

**Why add an Auditor?** Review finds bugs; Auditor catches the "tests pass but acceptance
spirit is missed" edge case. Risk-mitigation value justifies Fable's modest token cost.

**Why Read-only for Auditor?** Verification role should not modify code. If Auditor finds
an unmet criterion, it escalates back to Review (who has Write access) rather than trying
to fix it itself.

**Why Haiku for Write?** Docs updates are deterministic transformations (given facts from
Build/Review, write the facts into CLAUDE.md). Haiku is cheap and sufficient. Restricting
to Read/Edit on docs only removes accidental risk of overwriting code files.

---

### Critical Files Boundary Enforcement (Mandatory Checklist)

Every subagent MUST verify file boundaries before committing:

```
## Critical Files Boundary Validation
Before git commit:
1. Run: git status | grep -E 'modified:|new file:'
2. For EACH file, verify it's in the Critical Files list:
   [ ] src/adverse_score/<module>.py
   [ ] tests/unit/test_<module>.py
   [ ] tests/e2e/test_<module>_e2e.py
   (etc. — list all expected files)
3. If ANY file is outside the list:
   STOP → Report boundary violation (file path, rationale) to orchestrator
   Do NOT commit.

## Cross-Cutting Bug Escalation Protocol
If a subagent discovers a genuine bug OUTSIDE the Critical Files list:
1. Do NOT fix it (stays in scope boundary)
2. Create a message to orchestrator with:
   - File path and line number
   - Bug description (why it breaks the phase)
   - Why it's not in Critical Files
   - Suggested fix (if obvious) or note "requires separate phase"
3. Pause execution; await instruction
```

---

**NOTE:** The goal of multi-agent orchestration is to optimize token usage and prevent
scope creep or overlapping edits. Clear Critical Files boundaries, structured handoffs,
and read-only verification roles (Auditor, Write) all enforce these constraints. All tools
are available to each subagent as needed; the per-subagent "Tool access" lines define
scope boundaries, not tool restrictions — these are independent constraints.




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
- **Phase 8 Guardrail Enforcement: Hybrid Model (Prompt-Enforced vs. Code-Enforced)** — Phase 8 implements a deliberate hybrid approach to the seven guardrails, recognizing that some guardrails are fundamentally semantic judgments (not code-checkable) while others benefit from structural, deterministic enforcement: (1) Guardrails 1 (Causality Lock) and 7 (Scope Enforcement) are **system-prompt-enforced** — the system prompt explicitly instructs the LLM not to make causal claims and to decline/redirect out-of-scope requests. Guardrail 1 has a log-only defense-in-depth function `_flag_causal_language()` (scans response for causal-language patterns, logs warning, never blocks) to provide an audit trail without false-positive suppression. (2) Guardrails 4 (Human Review Requirement), 5 (Completeness & Methodology Transparency), and 6 (Formula Version Audit Trail) are **code-enforced, deterministic** — `_ensure_human_review_reminder()` and `_ensure_completeness_statement()` functions are applied unconditionally at turn-end whenever a dataset is in session, appending fixed disclosure text sourced directly from real `ConsolidationResult` fields (retrieved/estimated counts, truncation status, formula version) if the LLM's own response doesn't already include them. This design guarantees these three guardrails are "always present" structurally, independent of LLM memory. (3) Guardrails 2 (No-Fabrication Placeholder) and 3 (Mandatory Draft Marking) are **document-level guardrails, deferred entirely to Phase 9** (document generation) since they operate on `.docx` structure and content, not conversation flow. The system prompt is explicitly instructed not to reference document/export capabilities. Why this hybrid design? Semantic judgment (causality, scope) is best encoded in prompts but needs audit logging; structural guarantees (completeness disclosure, formula version) must be code-enforced; and document-level markup belongs in the generation layer, not conversation orchestration. This layering prevents over-engineering in one layer while ensuring high-assurance guardrails operate deterministically where appropriate.
- **Phase 7 Persistence: Chunk-Report-Trim Exception & Always-Insert-Never-Upsert** — The `persistence.py` module makes two deliberate departures from default patterns: (1) `PSURRetrievalResult.chunks[].reports` (pre-merge per-chunk lists) are trimmed to `[]` on save, breaking strict round-trip fidelity — this is a **documented exception** because the per-chunk data is a complete subset already folded into merged `retrieval.reports` (Phase 2) and deduplicated `dedup.reports` (Phase 3), so verbatim storage would duplicate report data ~2–3x for high-volume drugs, inflating SQLite size with no new information. Full audit trail remains (chunk metadata: label, date range, counts, truncation flag) so completeness/methodology transparency (Guardrail 5) is preserved. (2) `save_consolidation()` inserts a NEW row every time, never upserts/overwrites — this was a deliberate choice so a PV analyst re-running a consolidation on the same drug+period never silently loses a prior dataset they may have already started reviewing. Full history is retained; downstream UI can provide "resume a prior session" functionality via `list_consolidations()` and `load_consolidation()`. Combined, these choices optimize storage efficiency (rare but important), audit transparency, and analyst workflow (no accidental data loss).
- **Phase 9 Document Generation: Verified ICH E2C(R2) PBRER Structure & Trimmed-Subset Design** — Phase 9's `.docx` export uses the real verified ICH E2C(R2) PBRER structure (20 numbered sections + appendices), confirmed by direct reference to the official ICH guideline PDF. The design choice to populate only Sections 1, 6, 15, and 16.1–16.3 with real data (and all remaining sections via one consolidated placeholder note) is deliberate and justified: Sections 1 (Introduction) and 6 (Data in Summary Tabulations) capture canonical drug identity and retrieval/dedup methodology. Section 15 (Overview of Signals) and Section 16.1–16.3 (Signal Evaluation — with emphasis on evidentiary input requiring clinical validation, not completed risk characterization) consume the Phase 1–5 ranking output. Sections 16.4–16.5 (Characterization of Risks, Risk Minimisation) require exposure-denominator data outside this tool's scope. Sections 2–5, 7–14, and 16.4–19, 20 are organizational/background matter not dependent on FAERS consolidation. The guideline itself states a disproportionality statistic (PRR-based ranking) is not a validated signal — validation requires clinical/epidemiological expertise absent here. This architecture ensures the document honestly frames the Phase 1–5 output as evidentiary input to expert judgment (Guardrail framing), never as a finished clinical evaluation. The placeholder text is fixed, literal, never LLM-touched (Guardrail 2), enforcing regulatory honesty over narrative flow.

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
│   ├── test_persistence.py       # ConsolidationStore persistence, round-trip fidelity, conversation history (Phase 7)
│   ├── test_orchestrator.py      # orchestrator.py guardrail post-processing, causal flagging, tool-calling flow with mocked LLM, dataset carry-forward, iteration-cap enforcement (Phase 8)
│   ├── test_agent_tools.py       # agent_tools.py tool wrapping, JSON serialization, error surfacing, contextvars capture box (Phase 8)
│   ├── test_drug_identity.py     # drug_identity.py resolution, error handling (Phase 1)
│   ├── test_consolidation.py     # consolidation.py pipeline wiring, error handling, end-to-end integration (Phase 6)
│   └── test_document_generator.py # document_generator.py PBRER section structure, placeholder text, draft marking, completeness data, guardrail functions (Phase 9)
└── e2e/
    ├── test_fda_client_e2e.py     # Live openFDA API contract validation
    ├── test_fda_client_psur_e2e.py # PSUR chunked retrieval against live API (Phase 2)
    ├── test_deduplication_e2e.py  # Deduplication against live PSUR retrieval results (Phase 3)
    ├── test_prr_e2e.py            # PRR against live FAERS data
    ├── test_drug_identity_e2e.py  # Drug identity resolution against live API (Phase 1)
    ├── test_consolidation_e2e.py  # Full pipeline consolidation against live FDA data (Phase 6)
    ├── test_orchestrator_e2e.py   # Multi-turn agent against real openFDA + OpenAI APIs, dataset reuse, off-topic decline (Phase 8)
    └── test_document_generator_e2e.py # Live document generation against real KEYTRUDA/6mo data with real OpenAI narration (Phase 9)
```

**Current counts:** 233 unit tests (all passing, no API keys), 20 E2E tests (all passing against live openFDA + OpenAI APIs, including Phase 2 PSUR retrieval, Phase 3 deduplication, Phase 6 consolidation, Phase 8 multi-turn agent, and Phase 9 document generation tests).

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
