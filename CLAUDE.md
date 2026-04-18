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
  - **Recency decay**: 1.0 if <90 days, 0.5 otherwise
  - **Score formula**: `min(100, mean(base_weight * label_penalty * decay) * 40)`
  - **PRR**: Wald 95% CI on log-transformed PRR; signal if CI lower bound > 1.0 and drug cases >= 3
  - **Benchmarking**: Discovers pharmacologic class via FDA count endpoint, finds top 3 peers, averages their scores
- **`src/adverse_score/config.py`** — Loads `.env`, validates both API keys are present (fail-fast).

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
