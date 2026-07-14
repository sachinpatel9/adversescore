# AdverseScore — PSUR Consolidation Rebuild: Scope & Build Plan

**Document purpose:** This is the authoritative scope contract for the next iteration of AdverseScore. It replaces the prior five-capability architecture (Score Explainability, Unlabeled Signal Detection, Signal Narrative Generator, Temporal Trend Analysis, Comparative Scorecard) with a single, clearly defined agent job. This document should be treated as the source of truth for design decisions — where the codebase and this document disagree, this document wins unless explicitly revised.

**How to use this document:** Build phases are ordered by dependency, not by importance. Do not begin a phase until the prior phase's acceptance criteria are met and its tests pass. Each phase produces a working, tested unit of the system — this is intentional, to minimize compounding bugs across a large rebuild.

---

## 1. Why This Rebuild Exists

The original AdverseScore was built iteratively, one feature at a time, without a single unifying agentic design pass. It produced five parallel capabilities that each worked in isolation but did not represent one coherent job. This rebuild applies the **AGENT Blueprint framework** (Assignment, Guidance, Engine, Needs, Memory) to redefine AdverseScore around one real, specific pharmacovigilance pain point, and to bring the codebase in line with production-grade agentic design practices.

---

## 2. The Problem Being Solved

When a pharmacovigilance team prepares a **Periodic Safety Update Report (PSUR)** — internationally standardized as the **PBRER** under **ICH E2C(R2)** — they must consolidate all adverse event data for a drug across a defined reporting period, deduplicate case reports, prioritize the resulting signals, and produce a structured document that feeds the regulatory submission. Today this is manual: querying FAERS, exporting, deduplicating by hand, and assembling a narrative — a process that consumes significant PV analyst time per reporting cycle.

**AdverseScore's new job:** given a drug and a PSUR reporting period, consolidate, deduplicate, and rank all relevant FAERS adverse event signals, and generate a structured draft document aligned to the real PBRER format — ready for a PV analyst to review, edit, and carry forward.

---

## 3. AGENT Blueprint Summary

### 3.1 Assignment

**The agent's job, in one sentence:** Given a drug name and a PSUR reporting period, consolidate all FAERS adverse event reports for that drug within that period, deduplicate them, rank the resulting signals by clinical priority, and produce a PBRER-structured draft document.

**In scope:**
- Single-drug PSUR consolidation for a user-specified period
- Deduplication of FAERS case reports
- Deterministic, ranked signal prioritization (not a single composite score)
- Structured `.docx` export aligned to ICH E2C(R2) PBRER format
- Multi-turn conversational follow-up on an already-consolidated dataset
- Long-term memory of consolidated datasets and conversation history across sessions

**Explicitly out of scope:**
- Period-over-period comparison (e.g., "is this worse than last PSUR")
- A single composite "AdverseScore" risk number
- The five prior parallel capabilities as independent features
- Multi-drug or combination-product PSURs
- Full regulatory completeness of all PBRER sections (only signal-related sections are populated with real data; others are explicit placeholders — see Section 4, Guardrail 2)

**Signal ranking criteria** (all four applied together, deterministically, per signal):

| Criterion | What it measures | Primary data inputs |
|---|---|---|
| Seriousness & Outcome | Severity of the adverse event | FAERS seriousness flags, outcome codes (death, hospitalization, disability, etc.) |
| Strength of Evidence | Frequency and statistical signal strength | Report count, PRR + 95% CI (reuse existing `prr.py` math), **modified by label status** — an unlabeled signal carries more evidentiary weight than a labeled one at equivalent report volume |
| Reversibility | Whether the adverse event resolves upon discontinuation | Outcome/resolution fields where available; where FAERS data is insufficient, a documented reference heuristic is required (see Known Assumptions, Section 7) |
| Public Health Impact | Population exposure, underlying disease severity, preventability | Report volume as a directional proxy for exposure (documented limitation — see Section 7), therapeutic class context |

The **ranking itself is a deterministic Python calculation** — the same architectural pattern already used for PRR/CI math in the current codebase. **The LLM's role is exclusively to narrate and explain the ranking output** — it does not compute, reweigh, or override the ranking.

---

### 3.2 Guidance (Prompts)

- The system prompt must assume a **multi-turn conversational task**, not a single-shot query. A PV analyst may ask follow-up questions ("show me just the serious ones," "what about the unlabeled ones") that reference the **already-consolidated in-session dataset** — the agent must not re-run FDA retrieval for these follow-ups.
- The system prompt must encode all seven guardrails in Section 4 as non-negotiable behavioral constraints, not suggestions.
- The system prompt must explicitly state the agent's scope boundary (PSUR consolidation only) and redirect any off-topic request (general medical advice, unrelated drug questions, prescribing guidance).

---

### 3.3 Engine

**GPT-4o.** No engine change from the current implementation. Local/open-weight model support (Llama, Mistral, Ollama) is explicitly **not a goal for this iteration** and should not be built as a configurable option — this keeps the rebuild focused on agent job, tool, and memory architecture rather than introducing model-quality variables on top of an already substantial scope change.

**Known constraint to design around:** consolidated datasets for high-volume drugs across long periods (e.g., 3 years) may produce a large number of distinct signals. The narration step must not attempt to narrate an unbounded number of signals in a single LLM call — cap narrated detail to a reasonable top-N (e.g., top 20 ranked signals) in conversational responses, while the full ranked list is always present in the exported document.

---

### 3.4 Needs (Tools & Data)

- **openFDA FAERS API** remains the core data source.
- **Drug identity resolution** (new): FAERS drug names are free-text and inconsistent (brand name, generic name, misspellings, dosage-form suffixes). Before retrieval, the agent must resolve a user-provided drug name into its canonical brand/generic name variants using openFDA's own label/NDC data, so consolidation is not silently incomplete due to name-matching gaps. This must also resolve the drug's **market authorization/approval date**, since PSUR periods are anchored to it (see below).
- **Chunked, paginated retrieval** (modified): PSUR periods must be broken into sub-period chunks (e.g., quarterly) and queried separately, then merged. Each individual chunk query must stay within openFDA's 25,000-record skip ceiling. This is a correctness requirement, not an optimization — a naive single-query approach will silently truncate data for high-volume drugs over long periods.
- **Data completeness transparency is mandatory on every response and in the exported document** — retrieved count vs. estimated total count, stated even when no truncation occurred. If any individual chunk hits the ceiling, that specific sub-period must be flagged as potentially incomplete, both in conversation and in the document.
- **Deduplication** (new): primary matching on `safetyreportid`; fallback heuristic matching (same normalized drug name + same MedDRA event term + same event date) for cases where ID-based matching fails to catch true duplicates. Deduplication statistics (count removed, method used) must be retained and surfaced.
- **Label status classification** (retained, repurposed): still classifies each event as LABELED or UNLABELED against current FDA label text, but is no longer a standalone feature — it now feeds directly into the Strength of Evidence and Public Health Impact ranking criteria as a weighting input.
- **Document generation** (new): produces a `.docx` file. Requires a new dependency (`python-docx` or equivalent). Structure must map to real **ICH E2C(R2) PBRER section headers** — populate signal-related sections with real consolidated data; all other sections receive a literal, non-generated placeholder string (see Guardrail 2).
- **Tool boundaries are not pre-specified.** Claude Code should propose the tool decomposition during Phase 8 (Section 6), based on the module structure defined in Section 5. As a general principle, prefer single-responsibility tools over one large multi-purpose tool, consistent with the modular pattern already established in the current codebase (`fda_client.py`, `prr.py`, `scoring.py` as separate concerns).
- **All prior feature-specific code tied to the five old capabilities must be fully removed**, not archived or left dormant — see Phase 0 in Section 6.

---

### 3.5 Memory

**Short-term (in-session working memory):**
- Conversation history within a session, so follow-up questions can reference the already-consolidated dataset without re-querying FAERS.

**Long-term (persistent, cross-session memory):**
- The full consolidated, deduplicated, ranked dataset per drug + period, so reopening a prior PSUR consolidation does not require re-querying FDA.
- Conversation history, to support resuming a follow-up discussion in a later session.

**Explicitly not persisted:**
- User preferences/settings.
- Prior-period comparison data (out of scope per Section 3.1).

**Explicit assumption:** this is a single-analyst, local-first tool. No concurrent multi-user access handling is required or should be built.

---

## 4. Guardrail Framework (Rebuilt for This Job)

The shift from "informational chat response" to "artifact structurally resembling a regulatory document" raises the stakes on fabrication, provenance, and review requirements. These seven guardrails are rebuilt from first principles for this job and must be treated as hard constraints, not soft preferences.

**Guardrail 1 — Causality Lock**
The agent must never assert or imply a causal relationship between a drug and an adverse event. Signal ranking reflects statistical association and signal strength — never causation. The agent never recommends medication changes, dosing adjustments, or patient-level clinical action. This is a stricter, more precisely scoped replacement for the old "diagnosis lock," because "signal" and "cause" are easily and dangerously conflated in PV communication.

**Guardrail 2 — No-Fabrication Placeholder Rule**
For any ICH E2C(R2) PBRER section the tool does not populate with real consolidated data, the document must insert a **fixed, literal, non-LLM-generated placeholder string** (e.g., `[Section not populated by AdverseScore — to be completed by Regulatory Affairs]`). The LLM must never generate prose to fill a section it lacks real data for.

**Guardrail 3 — Mandatory Document-Level Draft Marking**
Every exported `.docx` must carry a prominent, structurally embedded marking — in a cover section and in the running header/footer — stating clearly that the document is a draft, not for regulatory submission, and pending qualified PV/clinical review. This must live in the document itself, not only in the conversation, because the document travels independently of the chat that produced it.

**Guardrail 4 — Universal Human Review Requirement**
Every generated PSUR document requires human clinical/regulatory sign-off before any use, regardless of what the ranking shows. This is a broadening from the old system's risk-tiered escalation (which only flagged high scores) — the artifact class itself now demands review, not just high-risk content within it.

**Guardrail 5 — Completeness & Methodology Transparency Embedded in the Document**
Retrieved-vs-total record counts, any per-chunk truncation flags, and the deduplication methodology used must appear as a stated section inside the exported document, not only spoken in conversation. A PV professional reading this document weeks later, without the original conversation, must be able to independently understand what data went in and what might be missing.

**Guardrail 6 — Ranking Formula Version Audit Trail**
The document must record which version of the deterministic ranking formula (weights for the four criteria) produced the results, supporting the same auditability principle the original `config.py` design was built around.

**Guardrail 7 — Scope Enforcement**
The agent declines and redirects any request outside PSUR consolidation — general medical advice, prescribing guidance, unrelated drug questions — consistent with the prior system's scope enforcement, re-scoped to this narrower job.

---

## 5. Target Module Structure

```text
adversescore/
├── app.py                          # Streamlit UI: PSUR period selector, ranked signal display, .docx export, chat
├── src/
│   └── adverse_score/
│       ├── config.py                # Ranking weights, PSUR period constants, dedup/chunking parameters, formula version
│       ├── drug_identity.py         # NEW — brand/generic name resolution, market authorization date lookup
│       ├── fda_client.py            # MODIFIED — chunked/paginated retrieval anchored to market authorization date
│       ├── deduplication.py         # NEW — safetyreportid matching + fallback heuristic
│       ├── label_classifier.py      # MODIFIED — now a weighting input to ranking, not a standalone feature
│       ├── prr.py                   # RETAINED — PRR + Wald 95% CI math, feeds Strength of Evidence
│       ├── ranking.py               # NEW (replaces scoring.py) — deterministic 4-criteria ranking engine
│       ├── consolidation.py         # NEW — orchestrates identity resolution → retrieval → dedup → labeling → ranking
│       ├── document_generator.py    # NEW — ICH E2C(R2) PBRER .docx generation with guardrails 2, 3, 5, 6 embedded
│       ├── agent_tools.py           # REBUILT — tool boundaries proposed by Claude Code in Phase 8
│       ├── orchestrator.py          # REBUILT — new system prompt, multi-turn guidance, all 7 guardrails
│       ├── persistence.py           # MODIFIED — new schema: consolidated datasets + conversation history
│       └── logger.py                # RETAINED — structured JSON logging, unchanged
├── data/                            # SQLite DB (auto-created, gitignored)
├── docs/
│   └── PSUR_CONSOLIDATION_SCOPE.md  # this document
├── conftest.py
├── test_adversescore.py             # REBUILT unit test suite
├── test_e2e.py                      # REBUILT E2E test suite (live API)
├── pytest.ini
├── requirements.txt                 # + python-docx (or equivalent docx library)
└── .env
```

---

## 6. Phased Build Plan

Each phase must be fully working and tested before the next phase begins. Do not skip ahead.

---

### Phase 0 — Foundation Cleanup

**Goal:** Remove all code tied to the five prior capabilities before adding anything new.

**Remove:**
- Old `scoring.py` composite-score logic (severity weights, label penalty multipliers feeding a single score)
- Old `config.py` constants tied to the composite score model (`SEVERITY_WEIGHT_*`, `LABEL_PENALTY_*`, `HUMAN_REVIEW_THRESHOLD`, `SPECIALIST_ROUTING_THRESHOLD`)
- Old `orchestrator.py` system prompt and routing logic
- Old `app.py` UI sections: sidebar history panel, portfolio scorecard, trend chart, five-capability chat flow
- Old `persistence.py` schema (single-analysis-per-row model)
- Old test cases tied to any of the above

**Acceptance criteria:** Repository builds and runs with all prior feature-specific code fully removed. No orphaned imports. No dead code referencing removed modules. `pytest` runs clean (even if the suite is now nearly empty — that's expected and correct at this stage).

---

### Phase 1 — Drug Identity Resolution

**Goal:** Given a raw user-provided drug name, resolve it to the canonical set of brand/generic name variants to query, and resolve the drug's market authorization/approval date.

**Build:** `drug_identity.py`
- Function to resolve a raw drug name input into all relevant brand/generic name variants, using openFDA's own label/NDC data.
- Function to look up the drug's market authorization/approval date (needed to anchor the PSUR period in Phase 2).
- Graceful handling of misspellings and unknown/not-found drugs — return a clear, structured error, not a silent empty result.

**Tests:**
- Known brand/generic pairs resolve to the same canonical identity.
- A common misspelling still resolves correctly.
- An unrecognized drug name returns a clear "not found" result rather than failing silently or crashing.

---

### Phase 2 — Chunked, Paginated FDA Retrieval

**Goal:** Retrieve all FAERS reports for a resolved drug identity across a PSUR period anchored to the market authorization date, safely within openFDA's pagination limits.

**Build:** modify `fda_client.py`
- Accept a period selection (6mo/1yr/2yr/3yr) and the market authorization date from Phase 1; compute the actual date range.
- Break the date range into sub-period chunks (e.g., quarterly) and query each separately.
- Paginate within each chunk, staying under the 25,000-record skip ceiling.
- Merge results across chunks without introducing boundary duplicates.
- Track and return completeness metadata: retrieved count vs. estimated total count per chunk, and a flag for any chunk that hit the ceiling.

**Tests:**
- Chunk boundaries are correctly computed for each period length.
- A chunk that hits the ceiling is correctly flagged as potentially truncated.
- Merging does not duplicate reports that fall on chunk boundaries.
- Existing retry/backoff behavior (urllib3 + tenacity) still functions correctly per chunk.

---

### Phase 3 — Deduplication

**Goal:** Remove duplicate case reports from the merged, chunked retrieval output.

**Build:** `deduplication.py`
- Primary matching on `safetyreportid`.
- Fallback heuristic matching (same normalized drug name + same MedDRA event term + same event date) for cases where ID-based matching fails to catch true duplicates.
- Return deduplication statistics: count removed, and which method (ID match vs. heuristic) caught each duplicate.

**Tests:**
- Exact `safetyreportid` duplicates are correctly removed.
- The fallback heuristic catches near-duplicate cases without over-merging genuinely distinct reports.
- Deduplication statistics accurately reflect what was removed and how.

---

### Phase 4 — Label Status Classification

**Goal:** Classify each deduplicated report's adverse event as LABELED or UNLABELED, to feed into ranking.

**Build:** modify `label_classifier.py`
- Reuse/adapt the existing FDA label text classification approach.
- Ensure output is structured to be consumed as a ranking input (Phase 5), not as a standalone user-facing feature.

**Tests:**
- A known labeled event classifies as LABELED.
- A known or simulated novel event classifies as UNLABELED.
- Missing or ambiguous label data is handled gracefully (does not crash the pipeline; flags as unknown rather than guessing).

---

### Phase 5 — Deterministic Ranking Engine

**Goal:** Rank deduplicated, labeled signals using the four criteria, deterministically.

**Build:** `ranking.py` (replaces `scoring.py`)
- Implement scoring for each of the four criteria (Seriousness & Outcome, Strength of Evidence, Reversibility, Public Health Impact) as defined in Section 3.1.
- Strength of Evidence and Public Health Impact must incorporate label status as a weighting modifier (unlabeled signals weighted higher at equivalent report volume).
- Combine into a final ranked signal list, with each signal's per-criterion breakdown preserved and exposed (not just a final rank number).
- Tag output with the current ranking formula version (Guardrail 6).

**Tests:**
- A high-severity, high-frequency signal ranks above a low-severity, low-frequency signal.
- An unlabeled event ranks above an otherwise-identical labeled event.
- Formula version tag is present and correct in output.

---

### Phase 6 — Consolidation Orchestration

**Goal:** Wire Phases 1–5 into one coherent, callable pipeline.

**Build:** `consolidation.py`
- Single entry-point function taking a drug name and period, returning a fully structured result: ranked signals with per-criterion breakdown, completeness metadata, deduplication stats, label status breakdown, ranking formula version.

**Tests:**
- Full pipeline integration test using a real drug and a short period (e.g., 6 months), validating correctness end-to-end against live FDA data.

---

### Phase 7 — Persistence Layer Redesign

**Goal:** Persist full consolidation results and conversation history for cross-session memory.

**Build:** modify `persistence.py`
- New schema storing the full structured consolidation result per drug + period (not the old single-score-per-row model).
- Conversation history table to support session resumption.

**Tests:**
- Round-trip save/retrieve preserves the full structured result without data loss.
- Conversation history retrieval returns entries in correct order.

---

### Phase 8 — Agent Orchestration & Guardrails

**Goal:** Build the conversational agent layer around the Phase 1–7 pipeline, with all seven guardrails enforced.

**Build:**
- `orchestrator.py` — new system prompt reflecting the PSUR consolidation job (Section 3.1), multi-turn follow-up support that reuses in-session consolidated data rather than re-querying FDA, and all seven guardrails from Section 4 encoded as explicit behavioral rules.
- `agent_tools.py` — Claude Code proposes the tool boundary decomposition at this phase, wrapping the Phase 1–7 modules as callable tools. Prefer single-responsibility tools consistent with the existing modular pattern.

**Tests:**
- Guardrail enforcement tests: the agent never asserts causality; the mandatory disclaimer/review language is always present; completeness/methodology transparency is always stated.
- A follow-up query correctly reuses the in-session dataset instead of re-fetching from FDA.
- An off-topic query (e.g., general medical advice) is correctly declined and redirected per Guardrail 7.

---

### Phase 9 — Document Generation

**Goal:** Generate the `.docx` PBRER-aligned draft document.

**Build:** `document_generator.py`
- Map output sections to real ICH E2C(R2) PBRER section headers.
- Populate signal-related sections with real consolidated/ranked data.
- Insert the literal, fixed placeholder string (Guardrail 2) for any section not populated with real data.
- Embed the mandatory draft marking in a cover section and in the document header/footer (Guardrail 3).
- Embed completeness/methodology transparency (Guardrail 5) and ranking formula version (Guardrail 6) in the document body.

**Tests:**
- Document contains all required PBRER section headers.
- Placeholder text appears verbatim (not LLM-paraphrased) for unpopulated sections.
- Draft marking is present in both the header/footer and a cover section.
- Completeness data and formula version appear correctly in the document body.

---

### Phase 10 — UI Integration

**Goal:** Build the Streamlit interface around the full pipeline.

**Build:** rebuild `app.py`
- PSUR period selector (6mo/1yr/2yr/3yr, anchored to market authorization date resolved in Phase 1).
- Drug name input with identity resolution feedback shown to the user.
- Ranked signal display (top-N in conversation, full list understood to be in the exported document).
- Chat interface supporting multi-turn follow-up.
- `.docx` download control.

**Tests:** Manual QA checklist — automated UI testing is not required, but verify manually: period selection produces correct date ranges, drug resolution feedback displays correctly, follow-up questions don't trigger redundant FDA queries, document download produces a valid, openable `.docx` file.

---

### Phase 11 — Full Test Suite & End-to-End Validation

**Goal:** Comprehensive validation of the complete rebuilt system.

**Build:**
- Full unit test pass across all new and modified modules (Phases 1–9). Target meaningful coverage of new logic — especially deduplication and ranking, which are the highest-risk components for subtle bugs.
- E2E test pass against live FDA data using at least 2–3 real drugs across different period lengths (including at least one high-volume drug to exercise the chunking/ceiling logic).
- Manual guardrail audit: verify all seven guardrails behave correctly across a range of queries, including adversarial ones (e.g., a user directly asking "should I take this drug" must be declined per Guardrails 1 and 7).

**Acceptance criteria:** All automated tests pass. Manual guardrail audit confirms correct behavior. A full PSUR consolidation for a real drug can be run end-to-end, producing a valid, correctly structured `.docx` document.

---

## 7. Known Assumptions & Limitations (Document Explicitly, Do Not Silently Resolve)

- **Reversibility** cannot always be directly derived from FAERS structured fields alone. Where data is insufficient, a documented reference heuristic or simplified assumption is acceptable — but it must be stated explicitly in code comments and in the document's methodology section, not silently guessed by the LLM.
- **Public Health Impact / population exposure** is approximated using FAERS report volume as a directional proxy, not true epidemiological exposure data. This is a known simplification appropriate for a prototype and should be stated as such in the document's methodology section.
- **Label classification** uses substring-matching against FDA label text, a known limitation carried forward from the prior implementation. This may produce false positives/negatives and should be noted as a limitation, not resolved with a larger engineering effort in this iteration.
- **This is a single-analyst, local-first tool.** No concurrent multi-user access, authentication, or multi-tenant data isolation should be built.
- **GPT-4o context limits**: for very large consolidated datasets, cap narrated conversational detail to a reasonable top-N of ranked signals; the full ranked list always lives in the exported document regardless of conversational cap.

---

## 8. Definition of Done for This Iteration

The rebuild is complete when: a PV analyst can provide a drug name and a PSUR period, receive a consolidated, deduplicated, ranked signal list with full completeness transparency, ask natural follow-up questions against that same dataset without redundant FDA queries, and export a PBRER-structured `.docx` draft document — all while every one of the seven guardrails in Section 4 holds true, and the full automated test suite passes.
