# AdverseScore

**An agentic pharmacovigilance system that consolidates FDA adverse event data into structured, audit-ready draft safety reports.**

Give AdverseScore a drug name and a reporting period. It resolves the drug's true identity, retrieves the *complete* set of FDA Adverse Event Reporting System (FAERS) case reports for that period (not a sample), deduplicates them, ranks every distinct safety signal deterministically across four clinical criteria, and exports a structured `.docx` document aligned to the ICH E2C(R2) Periodic Benefit-Risk Evaluation Report (PBRER) format — with every claim traceable back to real retrieved data.

It is built for pharmacovigilance (PV) analysts and safety teams who need to turn raw FAERS data into a defensible starting point for a Periodic Safety Update Report (PSUR), without waiting on a manual, spreadsheet-driven consolidation pass — and without the black-box risk of an AI system quietly inventing a number and calling it "risk."

> **Status:** Core pipeline, conversational agent, and document export are built and live-tested end to end across multiple drugs and reporting periods. All outputs are draft-marked and require sign-off by a qualified clinical/regulatory professional before any downstream use — see [Disclaimer](#disclaimer).

---

## Why AdverseScore

Pharmacovigilance signal detection today typically means an analyst manually pulling FAERS data, wrestling with pagination limits, deduplicating case reports by hand, and applying inconsistent judgment calls about what counts as "serious." It's slow, hard to reproduce, and hard to audit after the fact.

The obvious AI-era shortcut — feed the raw data to an LLM and ask for a risk score — is the wrong answer in a regulated domain. A single opaque number that mixes seriousness, evidence strength, and volume into one score is not defensible to a regulator, and an LLM asked to "just tell me the risk" will happily state things with more confidence than the underlying data supports.

AdverseScore takes a different position: **deterministic math does the ranking, the LLM only narrates it.** Every signal's position in the output is fully explainable by re-running the same fixed formula against the same data — no LLM call is in that loop at all. The LLM's job is strictly downstream: summarizing what the math already decided, in plain language, with guardrails that prevent it from asserting causation or inventing findings not present in the data.

---

## Core Capabilities

- **Canonical drug identity resolution** — resolves misspellings, brand names, and generic names to a single verified identity, including the drug's true market authorization date, via openFDA's own label/NDC/`drugsfda.json` data.
- **Complete FAERS retrieval, not a sample** — retrieves every matching case report across a full PSUR reporting period (6mo / 1yr / 2yr / 3yr), safely navigating openFDA's per-query pagination ceiling via automatic chunking, with explicit truncation flags if any window is genuinely too large to fully retrieve.
- **Two-pass deduplication with a full audit trail** — exact case-ID matching (version-aware) plus a deliberately conservative heuristic fallback pass, so near-duplicate reports don't inflate signal counts, without ever risking over-merging genuinely distinct cases.
- **Deterministic, explainable signal ranking** — every unique adverse event is ranked across four independent, clinically meaningful tiers (see [Technical Innovations](#technical-innovations)) — with **no single composite risk score**, by design.
- **Conversational follow-up, grounded in the real dataset** — ask follow-up questions about a consolidation ("show me just the serious ones," "what's unlabeled?") without re-querying FDA; the agent reuses the already-retrieved data for the session.
- **Regulatory-format document export** — one click produces a draft `.docx` aligned to the real ICH E2C(R2) PBRER structure, with the full ranked signal list, methodology transparency, and mandatory draft marking embedded directly in the file.
- **Cross-session memory** — every consolidation and every conversation turn is persisted, so an analyst can pick up a prior session days later without re-running retrieval.

---

## Technical Innovations

A few design decisions in this system are worth calling out specifically, because they're the parts that would be easy to get wrong in a regulated, LLM-adjacent product:

**1. Deterministic ranking with zero composite score.**
Signals are ranked by a strict, lexicographic four-tier sort — Seriousness & Outcome → Strength of Evidence → Reversibility → Public Health Impact — where each tier is a fully independent, human-readable classification (e.g. `DEATH`, `STRONG_UNLABELED`, `FATAL`, `HIGH`), not a weighted number folded into anything else. The sort key exists only internally; it is never stored or exposed. This means the *entire* ranking is reproducible by hand from the four published tier values — there is nothing hidden in a formula only the system can see.

**2. Real completeness, not a representative sample.**
Most integrations with the openFDA API accept its default page-limit sampling and quietly present a partial picture as if it were the whole one. AdverseScore instead anchors each PSUR reporting period to the drug's actual market authorization date (its International Birth Date, in ICH terms), slices that period into sub-windows, and paginates each window up to FDA's hard ceiling — explicitly flagging any window where even that wasn't enough, rather than silently truncating.

**3. A hybrid safety architecture for the LLM layer.**
Some safety properties (never asserting drug-caused-event causation, staying in scope) are semantic judgments that only a well-instructed LLM can make — so those are prompt-enforced with logged, non-blocking defense-in-depth monitoring. Other properties (disclosing data completeness, stating which ranking formula version produced this output, requiring human review) are non-negotiable and don't require judgment — so those are enforced deterministically in code, appended to every relevant response regardless of what the LLM said, sourced directly from the real underlying data. The system doesn't trust the LLM with guarantees code can make instead.

**4. A real ICH E2C(R2) PBRER structure, not an approximation.**
The exported document's section structure was built against the actual ICH guideline text (20 numbered sections, verified section-by-section), not a remembered or inferred approximation of it. Sections the system has real data for are populated; every other section carries a fixed, literal, non-LLM-generated placeholder — the model is structurally incapable of writing prose into a section it has no data for.

**5. Full retrieved dataset survives across a multi-turn conversation without re-fetching.**
A PV analyst's natural workflow is exploratory — "show me the serious ones," "what about unlabeled events?" The agent answers these from the dataset already in memory for that session rather than re-hitting FDA on every question, which matters both for cost/latency and for guaranteeing that follow-up answers are consistent with the first answer instead of drifting across separate live queries.

---

## System Architecture

```
Drug name + reporting period
        │
        ▼
┌─────────────────────┐
│ Identity Resolution  │  brand/generic name variants + market authorization date
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Chunked Retrieval   │  full-period FAERS pull, paginated, ceiling-flagged
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│   Deduplication      │  exact-ID + conservative heuristic pass, audit trail
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ Label Classification │  LABELED / UNLABELED against real FDA label text
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ Deterministic Ranking│  4-tier lexicographic sort, no composite score
└─────────┬────────────┘
          ▼
   ConsolidationResult ──────────────┬──────────────────┐
          │                          │                  │
          ▼                          ▼                  ▼
┌──────────────────┐      ┌──────────────────┐  ┌──────────────────┐
│  Persistence      │      │  Conversational   │  │  Document Export  │
│  (SQLite, full     │◄────┤  Agent (LangGraph, │  │  (.docx, ICH E2C   │
│  history retained) │     │  hybrid guardrails)│  │  (R2) PBRER format)│
└──────────────────┘      └──────────────────┘  └──────────────────┘
```

Each stage above is a separate, independently tested Python module with no hidden coupling — the ranking engine, for instance, has zero HTTP dependency and zero LLM dependency, so it can be verified as pure, deterministic math in isolation.

<details>
<summary><b>Full module reference</b></summary>

```text
adversescore/
├── app.py                          # Streamlit UI — period selector, identity/completeness cards,
│                                    # ranked signal table, chat, .docx export
├── src/adverse_score/
│   ├── config.py                   # Named constants: API config, retrieval/dedup/ranking tunables
│   ├── drug_identity.py            # Canonical name + market authorization date resolution
│   ├── fda_client.py               # openFDA client, IBD-anchored chunked/paginated retrieval
│   ├── deduplication.py            # Exact-ID + heuristic dedup, audit trail
│   ├── prr.py                      # Proportional Reporting Ratio + Wald 95% CI math
│   ├── label_classifier.py         # LABELED/UNLABELED classification vs. FDA label text
│   ├── ranking.py                  # Deterministic 4-tier signal ranking engine
│   ├── consolidation.py            # Single pipeline entry point (identity → ranking)
│   ├── persistence.py              # SQLite: consolidated datasets + conversation history
│   ├── agent_tools.py              # LangGraph tool wrappers around the pipeline
│   ├── orchestrator.py             # Multi-turn conversational agent + guardrail enforcement
│   ├── document_generator.py       # ICH E2C(R2) PBRER .docx generation
│   └── logger.py                   # Structured JSON logging
├── tests/
│   ├── unit/                       # 233 tests, no live API calls required
│   └── e2e/                        # 20 tests against live openFDA + OpenAI APIs
├── docs/
│   └── PSUR_CONSOLIDATION_SCOPE.md # Authoritative build specification
├── pytest.ini
└── requirements.txt
```

</details>

---

## Guardrails & Safety Architecture

Producing anything that resembles regulatory documentation demands controls well beyond a typical chat assistant. AdverseScore enforces seven guardrails, each mapped to how it's actually guaranteed:

| # | Guardrail | What it prevents | How it's enforced |
|---|---|---|---|
| 1 | **Causality Lock** | The system stating or implying a drug *caused* an event — only statistical association is ever claimed | Prompt-enforced, with logged pattern-matching as a monitoring layer |
| 2 | **No-Fabrication Placeholder** | The LLM writing plausible-sounding prose into a report section it has no real data for | Code-enforced — a fixed, literal, non-LLM-generated string is inserted structurally; the model never sees those sections |
| 3 | **Mandatory Draft Marking** | A generated document being mistaken for a final, submission-ready report | Code-enforced — embedded in the document's cover section *and* running header/footer, so it travels with the file even if separated from its source conversation |
| 4 | **Universal Human Review** | Any output being used without qualified sign-off, regardless of how the signals look | Code-enforced — appended deterministically to every response referencing a dataset |
| 5 | **Completeness & Methodology Transparency** | Silent, undisclosed data gaps (e.g. an over-large retrieval window) | Code-enforced — retrieved-vs-estimated counts and truncation status are sourced from real retrieval metadata, never LLM-stated |
| 6 | **Ranking Formula Version Audit Trail** | Losing the ability to reproduce or challenge a prior ranking after the formula changes | Code-enforced — every output is tagged with the exact formula version that produced it |
| 7 | **Scope Enforcement** | The system drifting into general medical advice, dosing guidance, or unrelated drug questions | Prompt-enforced, with logged pattern-matching as a monitoring layer |

Guardrails 1 and 7 rely on the LLM correctly following instructions — because whether something constitutes a causal claim or an out-of-scope request is a judgment call, not a string match. Guardrails 2 through 6 make no such assumption: they are structurally guaranteed by code that runs regardless of what the model outputs, sourced directly from the real underlying data. This split is deliberate — it puts the LLM's judgment where judgment is actually required, and puts hard guarantees everywhere else.

Full guardrail definitions and clinical rationale: `docs/PSUR_CONSOLIDATION_SCOPE.md`, Section 4.

---

## Getting Started

### Prerequisites
- Python 3.10+
- An [openFDA API key](https://open.fda.gov/apis/authentication/)
- An OpenAI API key (for the conversational agent and document narration)

### Setup

```bash
git clone https://github.com/sachinpatel9/adversescore.git
cd adversescore
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
OPENFDA_API_KEY=your_fda_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Run it

```bash
streamlit run app.py
```

Enter a drug name, pick a reporting period, and click **Run Consolidation**. From there: review the identity resolution and data completeness cards, browse the ranked signal table, ask follow-up questions in the chat, and generate a `.docx` export when ready.

### Run the tests

```bash
# Unit tests — fast, no API keys required
pytest tests/unit -v

# End-to-end tests against live openFDA + OpenAI APIs
pytest tests/e2e -v -m e2e

# Full suite
pytest -v
```

**Current status:** 233 unit tests, 20 E2E tests, all passing. A live consolidation of KEYTRUDA over a 6-month period retrieves and deduplicates several thousand real FAERS reports and ranks thousands of distinct signals end to end.

---

## Current Status

The full pipeline — identity resolution, chunked retrieval, deduplication, label classification, deterministic ranking, persistence, the conversational agent, document export, and the Streamlit UI — is built and fully tested end-to-end. Multi-drug live validation completed against KEYTRUDA (6-month and 2-year periods, exercising chunking at scale) and HUMIRA (6-month), all producing valid draft `.docx` exports. A structured adversarial guardrail audit confirmed enforcement of all seven safety guardrails, with findings documented in `docs/GUARDRAIL_AUDIT.md`; see `docs/PSUR_CONSOLIDATION_SCOPE.md` for the complete build specification and `docs/TODO.md` for detailed phase results.

<details>
<summary><b>Engineering build log (phase-by-phase detail)</b></summary>

### Phase 0: Foundation Cleanup
Removed all code tied to a prior five-capability system (composite scoring, narrative generation, temporal trend charts, portfolio scorecard, history panel), establishing a clean base for the PSUR consolidation rebuild.

### Phase 1: Drug Identity Resolution
Resolves a raw drug name input (e.g. "KEYTRUDA", "keytruda", "pembrolizumab") to canonical brand/generic name variants via openFDA label/NDC queries, plus market authorization date via `drug/drugsfda.json`. Returns a resolution confidence (`EXACT`, `FUZZY`, or `PARTIAL` — `PARTIAL` for OTC monograph drugs with no `drugsfda` entry) and a clear structured error if the drug isn't found at all.

### Phase 2: Chunked, Paginated FDA Retrieval
Retrieves the complete FAERS report set for a PSUR period, replacing a prior single-page "representative sample" approach. PSUR cycles are anchored to the drug's market authorization date (its International Birth Date), sliced into quarterly sub-chunks, and paginated up to openFDA's 25,000-record ceiling per chunk — with explicit truncation flagging and independent per-chunk error isolation. OTC drugs (no market authorization date) fall back to a rolling lookback window.

### Phase 3: Deduplication
Two-pass deduplication: exact `safetyreportid` matching with version awareness (highest `safetyreportversion` wins), then a deliberately conservative heuristic fallback pass requiring full-set equality on normalized drug names, symptom list, and receipt date — designed to never over-merge genuinely distinct multi-symptom cases. Full audit trail of every removal decision is retained.

### Phase 4: Label Status Classification
Classifies each unique symptom as `LABELED`, `UNLABELED`, or `LABEL_STATUS_UNKNOWN` against real FDA label text, batch-processed for the full deduplicated symptom set feeding into ranking.

### Phase 5: Deterministic Ranking Engine
Ranks every unique adverse event signal across four independent, lexicographically-ordered tiers — Seriousness & Outcome, Strength of Evidence, Reversibility, Public Health Impact — with no composite score ever computed or exposed. The internal sort key exists only to produce a stable order; it's never stored on output.

### Phase 6: Consolidation Orchestration
Single pipeline entry point wiring Phases 1–5 into one callable workflow, returning a fully structured result (ranked signals, completeness metadata, dedup statistics, label breakdown, formula version) or a structured error — never an unhandled exception. Live-tested against KEYTRUDA: 8,204 reports retrieved, 7,903 after dedup, 2,547 unique signals ranked.

### Phase 7: Persistence Layer
SQLite-backed cross-session memory. Every consolidation is inserted as a new row (never overwritten, preserving full history); conversation messages are stored per session with optional links to the consolidation they discuss. Full round-trip fidelity on save/load, with one documented, deliberate exception (pre-merge per-chunk report lists are trimmed on save since they're a strict subset of already-retained merged/deduplicated data).

### Phase 8: Agent Orchestration & Guardrails
A LangGraph-based multi-turn conversational agent (`run_agent_turn()`) reusing a session's already-retrieved dataset for follow-up questions instead of re-querying FDA. Implements the hybrid guardrail model described above. A `contextvars`-based capture mechanism carries the full structured result across LangGraph's internal tool-execution boundary (a genuine `threading.local()` limitation discovered and worked around during development).

### Phase 9: Document Generation
Generates `.docx` PSUR documents in the real, guideline-verified ICH E2C(R2) PBRER structure. Sections with real underlying data (Introduction, Data Summary Tabulations, Signal Overview, partial Signal Evaluation) are populated; every other section receives one fixed, literal, non-LLM-generated placeholder. Draft marking is embedded structurally in both the document header/footer and a cover-page paragraph.

### Phase 10: UI Integration
A Streamlit application wiring the full pipeline into a usable interface: drug/period controls that call the deterministic pipeline directly (the conversational agent is reserved for follow-ups only, never the initial data-gathering step), identity and completeness cards sourced from real data, a ranked signal table, a two-step document generation/download flow, and a "resume a prior session" feature. Live browser-tested end to end, including a real document download verified to contain the full uncapped signal list and structurally correct draft marking.

### Phase 11: Full Test Suite & End-to-End Validation
Comprehensive validation confirming all automated tests pass (236 unit, 20 E2E), both deduplication and ranking modules at 100% statement coverage. Multi-drug live validation completed: KEYTRUDA/6mo (8,204 retrieved, 2,547 ranked signals), KEYTRUDA/2yr (31,174 retrieved across quarterly chunks, 4,523 ranked signals — exercising chunking and pagination at scale), HUMIRA/6mo (3,668 retrieved, 2,052 signals). All three produced valid draft `.docx` exports verified by reopening and inspecting signal table row counts. A structured adversarial audit of all seven guardrails confirmed correct enforcement; findings and one deliberate-unfixed open item (multi-turn conversational fabrication) documented in `docs/GUARDRAIL_AUDIT.md`. One high-volume OTC drug (ASPIRIN) timed out during live testing and was deprioritized per product decision; this performance gap remains noted and scoped for future investigation but is non-blocking for the core rebuild.

</details>

---

## Disclaimer

AdverseScore produces **draft** pharmacovigilance signal consolidations and draft PBRER-aligned documents as evidentiary input for expert review — it does not perform, and is not a substitute for, clinical or regulatory signal evaluation. Every exported document is explicitly marked as a draft pending qualified clinical/regulatory review. AdverseScore is **not** a medical device and does **not** provide medical advice, causal safety determinations, or treatment recommendations. All outputs require validation by a qualified professional before any use in regulatory submissions or clinical decision-making.
