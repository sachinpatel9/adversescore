# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdverseScore is a clinical decision support agent that scores pharmaceutical safety signals using FDA adverse event data. It combines an openFDA API client, a statistical scoring engine, and a LangGraph-orchestrated GPT-4o agent behind a Streamlit chat UI.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run app.py
```

Requires a `.env` file with `OPENFDA_API_KEY` and `OPENAI_API_KEY` (see `.env.example`).

## Architecture

The execution flow is: **Streamlit UI → LangGraph agent → Pydantic validation → AdverseScoreClient → openFDA API → scoring math → JSON payload → LLM response**.

- **`app.py`** — Streamlit chat interface. Passes full conversation history to the agent for multi-turn support. Sets `recursion_limit=10` to prevent runaway agent loops.
- **`src/adverse_score/orchestrator.py`** — Wires the LangGraph react agent: GPT-4o (temperature=0.2), system prompt with safety protocols (SCOPE ENFORCEMENT, DIAGNOSIS LOCK, TOOL PROTOCOL), single tool binding.
- **`src/adverse_score/agent_tools.py`** — Defines `ClinicalQuerySchema` (Pydantic v2, `extra="forbid"`) and the `@tool`-decorated `get_adverse_score` function. The schema enforces strict types: `Literal["M","F"]` for sex, `ge=1,le=120` for age, `min_length=1` on drug/symptom strings.
- **`src/adverse_score/client.py`** — Core engine (~720 lines). Handles all openFDA API calls, scoring math, peer benchmarking, and PRR calculation. Key internals:
  - **Severity weights**: DEATH=1.75, HOSPITALIZATION=1.0, OTHER_SERIOUS=0.75, NON_SERIOUS=0.25
  - **Label penalty**: Unlabeled+Serious=2.0x, Unlabeled+Non-Serious=1.5x, Labeled=1.0x
  - **Recency decay** (3-tier): 1.0 if <90 days, 0.75 if 90-180 days, 0.5 if >180 days
  - **Score formula**: `min(100, mean(base_weight * label_penalty * decay) * 80)`
  - **Label matching**: Uses `re.search(r'\b' + re.escape(s) + r'\b', label_text)` word-boundary regex — not substring `in` — to avoid false positives on partial matches
  - **PRR**: Wald 95% CI on log-transformed PRR; signal if CI lower bound > 1.0 and drug cases >= 3
  - **Benchmarking**: Discovers pharmacologic class via FDA count endpoint, finds top 3 peers, averages their scores
- **`src/adverse_score/config.py`** — Loads `.env`, validates both API keys are present (fail-fast).

## Workflow Orchestration 

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something does not work, STOP and re-plan immediately - do not keep pushing forward
- Use plan mode for verification steps, not just building the codebase
- Write detailed specs upfront to reduce ambiguity 
- Use AskUserQuestion tool when appropriate to pressure test on all possible requirements for any updates or new additions to the codebase 

### 2. Subagent Strategy
- Use subagents liberally to manage context and keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, use more compute via subagents 
- Assign only one task per subagent for focused execution 

### 3. Self-Improvement Loop 
- Write rules for yourself that prevent the same mistakes and add them to CLAUDE.md as a reminder to yourself 

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this change?"
- Run tests, check logs, demonstrate accuracy

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask 'is there a more elegant way?'
- If a fix increases complexity of the codebase without a gain in efficiency and performance, then do not make the change
- Skip this for simple, obvious fixes - do not over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Big Fixing 
- When given a big report: just fix it. Do not ask for hand-holding 
- Point at logs, errors, and failing tests - then resolve them
- Zero context switching required from the user
- Fix failing CI tests without being told how
- If existing unit and E2E test files do not exist, write and maintain appending the files with new tests that are important to run as the codebase gets built but avoid redundancy 

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in with me before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards must be followed.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.








## Key Design Decisions

- All query-building methods must call `_sanitize_for_query()` before embedding values in Lucene strings. This is a security invariant — check it when adding new FDA queries.
- openFDA sex codes: **1=Male, 2=Female**. This was previously inverted and is a common source of bugs.
- The agent's error payload must include `clinical_disclaimer`, `diagnosis_lock`, `requires_human_review`, and `system_directive` — the system prompt rules depend on these fields being present in all payloads. The raw exception message must **never** appear in the payload; log it server-side via `print()` only.
- Peers with zero adverse event data are excluded from benchmark averages to prevent artificial score deflation.
- A pre-commit hook in `.git/hooks/pre-commit` blocks `.env` files and scans for API key patterns.
- **`AnalysisStore` (persistence.py)** supports the context manager protocol — prefer `with AnalysisStore() as store:` over manual `.close()` to guarantee connection cleanup on exceptions. `save_analysis` uses `.get()` for optional fields (`label_status`, `class_benchmark_avg`) and is safe to call on Incomplete Data payloads. `get_history` orders by `id DESC` (insertion order) so the most recently saved row is always first regardless of the timestamp value stored.
- **URL encoding in `client.py`**: `_discover_drug_class` and `_fetch_label_class_fallback` pass query parameters via `params=` dict to `session.get()` — do not use `urllib.parse.quote` manually. The `requests` library handles encoding. Drug/class names are still routed through `_sanitize_for_query()` before being embedded in Lucene field strings.
- **`_calculate_prr_metrics`** accepts optional `start_date` and `end_date` parameters and passes them through to `_fetch_symptom_counts`. Use these when computing time-bounded PRR (e.g. per-quarter temporal analysis).
- **Narrative drug name attribution** in `app.py`: after the agent returns, `st.session_state["last_analyzed_drug"]` is populated from the tool's JSON payload (`clinical_signal.drug_target`). The narrative save block uses this key — do not re-query `get_history(limit=1)` as that is a race condition in concurrent sessions.
- **`build_query` type hints**: `patient_age`, `patient_sex`, `start_date`, and `end_date` are all typed as `Optional[...]` — do not use bare `int = None` or `str = None` which suppress type checking.

## Test Suite

The project has two test files:

- **`test_adversescore.py`** — 153 unit tests covering Pydantic validation, query building, scoring math, PRR calculation, confidence metrics, guardrails, persistence, system prompt structure, narrative/temporal/delta protocols, and agent tool behavior. All tests use mocks — no API keys required. Runs in ~4 seconds.
- **`test_e2e.py`** — 29 end-to-end integration tests hitting the live openFDA API and OpenAI LLM. Covers FDA API contract validation, full pipeline scoring, agent tool invocation, LLM response quality (prose format, disclaimer, scope enforcement), and performance benchmarks. Requires API keys in `.env`; tests skip gracefully when keys are absent.

```bash
# Unit tests only (fast, no API keys)
python -m pytest test_adversescore.py -v

# E2E integration tests (requires .env)
python -m pytest test_e2e.py -v -m e2e

# Full suite
python -m pytest -v
```

For quick syntax validation without running full tests: `python -c "import ast; ast.parse(open('file').read())"`
