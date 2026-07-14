# AdverseScore: PSUR Consolidation Agent (Under Rebuild)

**AdverseScore** is an agentic pharmacovigilance tool for consolidating FAERS adverse event data into Periodic Safety Update Reports (PSURs). Given a drug name and reporting period, it resolves the drug's canonical identity, retrieves and deduplicates case reports, ranks signals deterministically, and exports structured `.docx` documents aligned to ICH E2C(R2) PBRER format. All outputs are marked as draft pending qualified clinical/regulatory review.

**⚠️ REBUILD IN PROGRESS:** The codebase is undergoing active refactoring per `docs/PSUR_CONSOLIDATION_SCOPE.md` (11-phase plan). **Phases 0–9 complete** (Foundation Cleanup, Drug Identity Resolution, Chunked FDA Retrieval, Deduplication, Label Status Classification, Deterministic Ranking Engine, Consolidation Orchestration, Persistence Layer Redesign, Agent Orchestration & Guardrails, Document Generation); **Phases 10–11 pending** (new UI, full validation). The Streamlit chat is currently a placeholder. The old five-capability system (Score Explainability, Signal Narrative Generator, Temporal Trend Analysis, Comparative Scorecard) has been removed and will **not** be rebuilt in this iteration.


## What's Built (Phase 0–9)

### Phase 0: Foundation Cleanup
Removed all code tied to the old five-capability system (composite scoring, narrative generation, temporal trend charts, portfolio scorecard, history panel). The repository is now clean slate for the PSUR consolidation rebuild.

### Phase 1: Drug Identity Resolution
Given a raw drug name input (e.g., "KEYTRUDA", "keytruda", "pembrolizumab"), the `drug_identity.py` module resolves it to:
- **Canonical brand/generic name variants** via openFDA label/NDC queries (exact match, then broadened)
- **Market authorization (FDA approval) date** via `drug/drugsfda.json` (earliest submission_status_date across matched NDAs/ANDAs/BLAs with approval status)
- **Resolution confidence** (`EXACT`, `FUZZY`, or `PARTIAL`) — `PARTIAL` when OTC monograph drugs are found (no drugsfda entry exists, but name variants are known)
- **Clear structured errors** if the drug is not found (not a silent empty result)

This bounds misspelling tolerance to openFDA's own Lucene search grammar; no custom fuzzy-matching library was added.

### Phase 2: Chunked, Paginated FDA Retrieval
Given resolved drug identity (name variants + market authorization date), retrieves ALL FAERS adverse event reports across a PSUR reporting period, safely within openFDA's pagination limits. Replaces the old single-page "500 record representative sample" approach:
- **IBD-anchored period computation** — PSUR cycles anchored to the drug's market authorization date (International Birth Date in ICH E2C(R2) terminology), selecting the most recently *completed* fixed-length cycle (6-month, 1-year, 2-year, or 3-year)
- **Quarterly chunking** — Slices PSUR period into sub-chunks (default 3-month windows), labeled with explicit date ranges (`chunk-{i}-YYYYMMDD-to-YYYYMMDD`)
- **Pagination with ceiling** — Paginates each chunk using `skip`/`limit`, capped at 25,000 (openFDA hard limit). Flags chunks as `truncated=True` if ceiling is hit; per-chunk errors don't abort other chunks
- **Boundary-collision dedup** — Version-aware merge on `report_id` across chunks to handle inclusive date-range boundaries (chunk i and i+1 may share an event_date if it lands exactly on the boundary); higher `safetyreportversion` wins on collision
- **OTC fallback** — For OTC monograph drugs (no market authorization date) or drugs approved less than one full period ago (no completed cycle yet), falls back to rolling lookback from today (flagged with reason)

Returns `PSURRetrievalResult` with per-chunk metadata (retrieved/estimated counts, truncation flags, optional per-chunk errors) and merged report list. No circular imports with `drug_identity.py` — accepts plain primitives (name variants list, optional date, period string).

### Phase 3: Deduplication
Given a flattened list of FAERS case reports (from Phase 2's `fetch_psur_reports()`), the `deduplication.py` module removes duplicates in two passes:
- **Exact `safetyreportid` matching with version awareness** — Groups reports by `report_id`, keeps the highest `safetyreportversion` per ID. Reports with missing/`None` `report_id` receive unique synthetic keys and never collapse into each other (defensive measure against malformed upstream data).
- **Fallback heuristic matching on survivors** — Two reports with *different* `report_id`s are treated as duplicates only if their full normalized `drug_names` set AND full normalized `symptom_list` set AND `receivedate` are ALL identical (deliberately strict to avoid over-merging genuinely distinct multi-symptom cases). Reports with empty `drug_names` or empty `symptom_list` are excluded from heuristic matching.

Returns `DedupResult` with original reports, input/output counts, per-method removal counts, and audit trail (list of removal decisions). Known limitation: `date` is `receivedate` (FDA receipt date), not a true per-reaction adverse-event onset date — FAERS lacks a single unambiguous event date, so this is the proxy used here and elsewhere in the codebase.

### Phase 4: Label Status Classification
Given a PSUR's merged, deduplicated symptom list and a drug's FDA label text, the `label_classifier.py` module classifies each symptom as LABELED, UNLABELED, or LABEL_STATUS_UNKNOWN (substring match on lowercased label). Phase 4 adds a batch processing function `classify_label_statuses()` alongside the existing single-symptom `classify_label_status()`, enabling efficient classification of all unique symptoms at once for Phase 5 ranking consumption. The batch function normalizes symptoms (uppercase, deduplicate case-insensitively, sort for determinism, filter empty), iterates once per unique symptom, and returns a frozen dataclass with per-symptom status mapping plus aggregate counts. The frozen dataclass prevents accidental mutation downstream; input symptom list is never modified.

### Phase 5: Deterministic Ranking Engine
Given deduplicated FAERS reports, outcome-code metadata, label classification results, and PRR metrics, the `ranking.py` module ranks all unique adverse event signals (MedDRA PT symptoms) across four lexicographically-ordered tiers:
- **Seriousness & Outcome** — DEATH > HOSPITALIZATION > OTHER_SERIOUS > NON_SERIOUS, derived from existing report severity fields
- **Strength of Evidence** — STRONG_UNLABELED > STRONG_LABELED > WEAK_UNLABELED > WEAK_LABELED > UNKNOWN_LABEL_STATUS, reusing PRR math with unlabeled signals outranking labeled ones at equivalent strength
- **Reversibility** — FATAL > POOR > REVERSIBLE > UNKNOWN, derived from newly-extracted per-reaction FAERS outcome codes (1–6)
- **Public Health Impact** — HIGH (100+ reports) > MODERATE (20–99 reports) > LOW (<20 reports), a proxy for exposure

Returns `RankingResult` containing a globally-ranked list of `RankedSignal` objects (each with tier names, PRR metrics, report count) plus label summary and formula version for audit trail. **Critical design:** The ranking uses an internal lexicographic sort key (tuple of tier ordinals) for determinism, but this key is never stored or exposed on output dataclasses — the four tier names are exposed as human-readable strings only. This design explicitly avoids the scope doc's ban on a single composite "AdverseScore" risk number while maintaining reproducibility and auditability.

### Phase 6: Consolidation Orchestration
Single pipeline entry point `consolidate_psur()` wires Phases 1–5 together into one coherent, callable workflow. Accepts a drug name and PSUR period, resolves the drug's identity, retrieves and deduplicates all FAERS reports across the period, classifies symptom labels, and ranks signals across four criteria. Returns `ConsolidationResult` with ranked signals (including per-criterion breakdown), completeness metadata (retrieved vs. estimated report counts, per-chunk truncation flags), deduplication statistics, label status breakdown, and ranking formula version for audit trail. Gracefully handles errors throughout the pipeline via structured `ConsolidationError` (never raises for domain failures). Tested end-to-end against live FAERS data: real consolidation of KEYTRUDA (8,204 reports retrieved, 7,903 after dedup, 2,547 unique signals ranked).

### Phase 7: Persistence Layer Redesign
SQLite-backed persistence enabling cross-session memory for multi-turn follow-ups without re-querying FDA. `ConsolidationStore` class (renamed from `AnalysisStore`) manages two tables: `consolidations` (canonical_name, period, result_json with full nested `ConsolidationResult`), indexed on (canonical_name, period, created_at DESC) for efficient "latest" lookup; `conversation_messages` (session_id, role, content, optional consolidation_id link), indexed on (session_id, id) and ordered by insertion order to avoid same-timestamp collisions. Serialization via `dataclasses.asdict()` with two deliberate post-processing steps: (1) `DrugIdentity.market_authorization_date` (date object) ↔ ISO string (None passes through for OTC drugs); (2) `PSURRetrievalResult.chunks[].reports` trimmed to `[]` on save — a documented exception to round-trip fidelity since this data is redundant (already in merged/deduplicated report lists) and would cause ~2–3x storage bloat for high-volume drugs. Deserialization reconstructs full `ConsolidationResult` objects (not plain dicts) indistinguishable from freshly-returned ones. New public methods: `save_consolidation()` (inserts new row every time, never overwrites — full history retained for audit trail), `load_consolidation()`, `get_latest_consolidation()`, `list_consolidations()` (lightweight metadata for "resume session" UI), `save_message()`, `get_conversation_history()`. Tests: 18 tests covering schema creation, full round-trip fidelity on real nested instances, None market-authorization-date handling, empty signals, multiple saves without overwriting, filtering, and session ordering.

### Phase 8: Agent Orchestration & Guardrails
LangGraph-based multi-turn conversational agent with embedded guardrails enforcing clinical/regulatory safety. Entry point: `run_agent_turn(conversation_history, user_message, in_session_dataset)` — a pure function returning `AgentTurnResult` (response_text, updated_dataset, dataset_changed flag). Orchestrator manages two agent tools: `resolve_drug_identity_tool()` for lightweight identity questions and `consolidate_psur_tool()` (primary data gathering, returns top 20 ranked signals + completeness metadata to avoid LLM context bloat). Novel technical pattern: contextvars-based capture mechanism (ContextVar-held per-call list) carries full `ConsolidationResult` across LangGraph's tool-node boundary (threading.local() doesn't work because nodes run on different worker thread). Guardrail enforcement uses hybrid model: Guardrails 1 (Causality Lock) & 7 (Scope Enforcement) are system-prompt-enforced with log-only defense-in-depth (`_flag_causal_language()` function); Guardrails 4/5/6 (Human Review Requirement, Completeness Transparency, Formula Version Audit Trail) are code-enforced unconditionally at turn-end via deterministic disclosure functions (sourced from real `ConsolidationResult` fields, never LLM-generated); Guardrails 2/3 (No-Fabrication Placeholder, Draft Marking) deferred to Phase 9 (document generation). Config additions: `OPENAI_CHAT_MODEL="gpt-4o"`, `AGENT_TEMPERATURE=0.1`, `AGENT_MAX_ITERATIONS=5`. Installed LangGraph 1.1.0, LangChain 1.2.11, langchain-openai 1.1.11. Multi-turn follow-ups reuse in-session dataset without re-fetching FDA. Tests: 38 unit tests + 2 E2E tests covering guardrail post-processing, causal-language flagging, tool-calling flow with mocked LLM, dataset carry-forward semantics, iteration-cap enforcement, no-persistence-import safety, and multi-turn dataset reuse against live APIs.

### Phase 9: Document Generation
Generates `.docx` PSUR documents in ICH E2C(R2) PBRER format (20 sections + appendices), verified against the official guideline PDF. Entry point: `generate_psur_document(result: ConsolidationResult) -> Document` returns a `python-docx` Document object for caller to save or stream. Trimmed-subset design: Sections 1 (Introduction: canonical name, brand/generic variants, market authorization date, therapeutic class, reporting period) and 6 (Data in Summary Tabulations: retrieved/estimated counts, per-chunk truncation flags, dedup methodology and stats) capture identity and completeness. Section 15 (Overview of Signals) contains the full ranked-signal table (uncapped, unlike Phase 8's 20-signal chat narration) with one LLM-narrated introductory paragraph. Sections 16.1–16.3 (Signal Evaluation) feature LLM-narrated text grounded in label-status classification and PRR-ranking data, explicitly framed as evidentiary input requiring clinical validation, not a completed evaluation. All remaining sections (2–5, 7–14, 16.4–19, 20/Appendices) use one consolidated literal placeholder note (Guardrail 2). Draft marking embedded structurally twice: docx header/footer objects AND cover-page body paragraph (Guardrail 3). Completeness metadata, ranking formula version, and mandatory human-review requirement appended deterministically (Guardrails 4, 5, 6). LLM narration via single `_narrate_signals()` call with `DOCUMENT_NARRATION_TEMPERATURE = 0.0` for faithfulness; fabrication guardrails include `_check_narration_for_fabricated_entities` (allowed-vocabulary scan) and reimplemented causal-language flagging (never depends on Phase 8 orchestrator). Two review-phase bugs found and fixed: (1) Section 20 gap in omitted-sections note covering (fixed by extending note, verified via full 1-20 accounting). (2) False-suppression bug on deterministic "evidentiary input, not completed" disclaimer (required both markers before skipping append, preventing accidental suppression). New config constants: `DOCUMENT_NARRATION_TEMPERATURE`, `PBRER_PLACEHOLDER_TEXT`, `PBRER_DRAFT_MARKING_TEXT`, `PBRER_OMITTED_SECTIONS_NOTE`. New dependency: `python-docx` 1.2.0. Tests: 34 unit tests covering section headers present, placeholder byte-exact, draft marking structural checks, completeness/formula version from real fixtures, full ranked-signal table, OTC/None date handling, empty signals, and guardrail functions tested directly. E2E: 1 live test with KEYTRUDA/6mo producing 2,548-row signal table in 84KB `.docx` file.


## Planned Architecture (Phases 10–11)

The full PSUR consolidation workflow (currently under build) will combine:

1. ✅ **Chunked FDA Retrieval** (Phase 2, DONE) — Break PSUR periods into quarterly chunks to stay within openFDA's 25K-record skip ceiling; retrieve and merge results, flagging any per-chunk truncation.
2. ✅ **Deduplication** (Phase 3, DONE) — Match case reports on exact `safetyreportid` with version awareness, with fallback heuristic matching (full-set equality on normalized drug names + symptom list + receipt date) for FAERS ID gaps. Audit trail of all removal decisions included in output.
3. ✅ **Label Status Classification** (Phase 4, DONE) — Batch classify symptoms as LABELED/UNLABELED/LABEL_STATUS_UNKNOWN via substring match against FDA label text. Normalize symptom input (uppercase, deduplicate, sort), return frozen dataclass with per-symptom status and aggregate counts.
4. ✅ **Signal Ranking** (Phase 5, DONE) — Deterministic 4-criteria ranking (Seriousness & Outcome, Strength of Evidence, Reversibility, Public Health Impact), using reusable PRR math + label-status weighting, with lexicographic tier sort and no composite score. The LLM's role is **only to narrate** the pre-computed ranking, not to compute or override it.
5. ✅ **Persistence Layer** (Phase 7, DONE) — SQLite schema for consolidated datasets + conversation history, enabling multi-turn follow-ups on cached data without re-querying FDA.
6. ✅ **Agent Orchestration & Guardrails** (Phase 8, DONE) — Multi-turn LangGraph agent with 7-guardrail system prompt (hybrid prompt + code enforcement), reusing in-session consolidated data for follow-ups.
7. ✅ **Document Generation** (Phase 9, DONE) — Produce `.docx` files with ICH E2C(R2) PBRER structure: signal-related sections populated with real data, all other sections filled with explicit placeholders (per Guardrail 2). Embedded guardrails: draft marking (Guardrail 3), completeness metadata (Guardrail 5), ranking formula version (Guardrail 6).
8. **New UI & Full Validation** (Phases 10–11) — Streamlit UI with PSUR period selector and ranked-signal display; comprehensive unit/E2E test pass and manual guardrail audit.

See `docs/PSUR_CONSOLIDATION_SCOPE.md` for the complete 11-phase spec including known limitations, assumptions, and implementation details.


## System Architecture (Current State)

```text
adversescore/
├── app.py                                 # Streamlit UI (placeholder, to be rebuilt Phases 10–11)
├── src/
│   └── adverse_score/
│       ├── config.py                      # API keys + named constants (Phase 1–2, 5, 8, 9 additions)
│       ├── drug_identity.py               # Drug resolution (Phase 1)
│       ├── fda_client.py                  # openFDA HTTP client + PSUR chunked retrieval (Phase 2, 5)
│       ├── deduplication.py               # Deduplication engine (Phase 3)
│       ├── ranking.py                     # Deterministic signal ranking engine (Phase 5)
│       ├── consolidation.py               # Consolidation pipeline entry point (Phase 6)
│       ├── client.py                      # Reduced orchestrator (removed scoring methods)
│       ├── prr.py                         # PRR + Wald 95% CI (unchanged)
│       ├── label_classifier.py            # Label classification only (removed penalty)
│       ├── agent_tools.py                 # LangGraph agent tools, contextvars capture box (Phase 8)
│       ├── orchestrator.py                # LangGraph multi-turn agent with guardrails (Phase 8)
│       ├── persistence.py                 # SQLite persistence + conversation history (Phase 7)
│       ├── document_generator.py          # `.docx` PBRER-aligned PSUR document generation (Phase 9)
│       └── logger.py                      # JSON-structured logging (unchanged)
├── data/                                  # SQLite DB (auto-created, gitignored)
├── docs/
│   └── PSUR_CONSOLIDATION_SCOPE.md        # Authoritative 11-phase rebuild spec
├── tests/
│   ├── conftest.py                        # Pytest fixtures
│   ├── unit/
│   │   ├── test_fda_client.py             # Includes Phase 2 PSUR chunking tests, Phase 3 flatten/merge tests
│   │   ├── test_deduplication.py          # Phase 3 deduplication tests
│   │   ├── test_prr.py
│   │   ├── test_label_classifier.py
│   │   ├── test_persistence.py
│   │   ├── test_orchestrator.py
│   │   ├── test_agent_tools.py
│   │   ├── test_drug_identity.py          # Phase 1
│   │   ├── test_consolidation.py          # Phase 6 pipeline integration tests
│   │   └── test_document_generator.py     # Phase 9 document generation tests
│   └── e2e/
│       ├── test_fda_client_e2e.py
│       ├── test_fda_client_psur_e2e.py    # Phase 2
│       ├── test_deduplication_e2e.py      # Phase 3
│       ├── test_prr_e2e.py
│       ├── test_drug_identity_e2e.py      # Phase 1
│       ├── test_consolidation_e2e.py      # Phase 6 end-to-end live FDA test
│       ├── test_orchestrator_e2e.py       # Phase 8 multi-turn agent test
│       └── test_document_generator_e2e.py # Phase 9 document generation E2E test
├── pytest.ini                             # pythonpath=src tests, testpaths=tests
├── requirements.txt                       # Dependencies
└── .env                                   # API keys (gitignored)
```


## Clinical Guardrails (New Framework)

The shift from "informational chat" to "artifact resembling regulatory documentation" demands stricter safety controls. The rebuilt system will enforce seven guardrails (to be implemented in Phase 8):

1. **Causality Lock** — No causal assertions between drug and adverse event; only statistical association and signal strength. No medication changes, dosing recommendations, or patient-level clinical action.
2. **No-Fabrication Placeholder Rule** — PBRER sections without real consolidated data receive fixed, literal, non-LLM-generated placeholders (e.g., `[Section not populated by AdverseScore — to be completed by Regulatory Affairs]`).
3. **Mandatory Document-Level Draft Marking** — Every exported `.docx` carries prominent, structurally embedded draft disclaimers in cover section and running headers.
4. **Universal Human Review Requirement** — All generated PSUR documents require clinical/regulatory sign-off before use, regardless of signal severity.
5. **Completeness & Methodology Transparency** — Retrieved-vs-total counts, per-chunk truncation flags, and deduplication method details are embedded in the exported document, not just spoken in conversation.
6. **Ranking Formula Version Audit Trail** — The document records which version of the deterministic ranking formula produced the results, supporting auditability.
7. **Scope Enforcement** — The agent declines and redirects any request outside PSUR consolidation (general medical advice, prescribing guidance, unrelated drugs).

See `docs/PSUR_CONSOLIDATION_SCOPE.md` Section 4 for full guardrail definitions and clinical rationale.


## Getting Started

### Prerequisites
* Python 3.10+
* openFDA API Key ([request here](https://open.fda.gov/apis/authentication/))
* OpenAI API Key (GPT-4o) — required only when Phase 8+ agent is available

### Installation
1. Clone the Repository
```bash
git clone https://github.com/sachinpatel9/adversescore.git
cd adversescore
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Configure Environment (for now, only openFDA key is required)

Create a `.env` file in the root directory:
```
OPENFDA_API_KEY=your_fda_key_here
# OPENAI_API_KEY=your_openai_key_here  (required in Phase 8+)
```

4. Running Tests (Current Recommendation)
Since the Streamlit UI is a placeholder, focus on running the test suite to validate the current Phase 0–5 work:

```bash
# Unit tests only (fast, no API keys required)
pytest tests/unit -v

# E2E integration tests (requires OPENFDA_API_KEY in .env)
pytest tests/e2e -v -m e2e

# Full suite
pytest -v
```

**Current test status:** 233 unit tests passing, 20 E2E tests (all passing against live openFDA + OpenAI APIs, including Phase 2 PSUR retrieval, Phase 3 deduplication, Phase 6 consolidation, Phase 7 persistence round-trip, Phase 8 multi-turn agent, and Phase 9 document generation tests).

5. Launching the Placeholder UI (Not Recommended Yet)
```bash
streamlit run app.py
```
The chat is a placeholder pending Phase 8 completion.

---

## Development Roadmap

- **Phase 0–1:** ✅ Foundation cleanup, drug identity resolution
- **Phase 2:** ✅ Chunked FDA retrieval
- **Phase 3:** ✅ Deduplication engine
- **Phase 4:** ✅ Label status classification
- **Phase 5:** ✅ Signal ranking (deterministic 4-criteria)
- **Phase 6:** ✅ Consolidation orchestration
- **Phase 7:** ✅ Persistence layer & conversation history
- **Phase 8:** ✅ Agent orchestration & guardrails
- **Phase 9:** ✅ `.docx` document generation
- **Phase 10–11:** Multi-turn UI, full test suite & end-to-end validation

See `docs/PSUR_CONSOLIDATION_SCOPE.md` for full details.

---

## Disclaimer

**AdverseScore is under active rebuild and currently non-functional as a user-facing tool.** This repository is shared for development transparency and team collaboration. When complete, AdverseScore will be a research-grade tool for generating draft PSUR documents from FAERS data. All outputs require validation by a qualified clinical/regulatory professional before use in regulatory submissions or clinical decisions. AdverseScore is **not** a medical device and does **not** constitute medical advice.
