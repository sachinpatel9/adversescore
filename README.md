# AdverseScore: Agentic Clinical Safety & Pharmacovigilance Engine

**AdverseScore** is an agentic clinical decision support platform that transforms raw FDA adverse event data into actionable pharmacovigilance intelligence. Built for clinicians and PV teams managing drug portfolios, it combines real-time openFDA querying, statistical disproportionality analysis (PRR with 95% CI), automated drug label classification, quarterly trend detection, ICH E2E-aligned narrative generation, and longitudinal portfolio tracking — all orchestrated by a LangGraph agent behind a Streamlit interface. AdverseScore is designed to reduce the manual burden of PSUR cycles and signal management by surfacing emerging risks, benchmarking drugs against therapeutic peers, and generating regulatory-ready draft documentation.


## Core Technical Innovations

### 1. Dynamic Ontology Mapping (Algorithmic Discovery)
Unlike traditional tools that rely on hardcoded drug dictionaries, AdverseScore implements a **Frequency-Based Discovery Engine**. It queries the openFDA database to mathematically identify the dominant Established Pharmacologic Class (EPC) for any therapeutic or over-the-counter medication. This allows the system to automatically discover and benchmark a drug against its true real-world peers (ex. Comparing KEYTRUDA against other PD-1 inhibitors) without manual maintenance.

### 2. Disproportionality Analysis (PRR & 95% CI)
At its core, the AI-powered system implements a **Proportional Reporting Ratio**.
- **Statistical Grounding:** Calculates the PRR to detect signals that are disproportionately frequent in a specific drug compared to its pharmacologic class
- **Confidence Intervals:** Computes the 95% Confidence Interval (CI) lower bound to ensure clinical signals are statistically significant ($a \ge 3$ and $CI_{lower} > 1.0$) before alerting the user.

### 3. Agentic Orchestration with LangGraph
Built on a **LangGraph** architecture, the system utilizes a state-aware agent to:
- **Natural Language Extraction:** Uses Pydantic schemas to isolate target symptoms and patient demographics (Age/Sex) from unstructured clinical prompts.
- **Deterministic Guardrails:** Controls LLM behavior through Python-level booleans. If the mathematical engine detects a high-risk signal, the agent is programmatically forced into a "Human Review" fallback state, bypassing its generative autonomy.


## Product Capabilities

### 1. Score Explainability
Every response includes a mandatory **Score Rationale** section explaining which adverse event signals most influenced the score, whether they are labeled or unlabeled, how far the PRR was above or below threshold, what the peer benchmark contributed, and any data confidence caveats.

### 2. Unlabeled Signal Detection
Automatically classifies adverse events as **LABELED** or **UNLABELED** by querying FDA drug labeling data. Applies severity-weighted label penalties to the AdverseScore calculation:
- Unlabeled + Serious outcome: **2.0x** weight multiplier
- Unlabeled + Non-Serious outcome: **1.5x** weight multiplier
- Labeled events: **1.0x** (no penalty)

### 3. Signal Narrative Generator
Generates **ICH E2E-structured** signal evaluation narrative drafts triggered by documentation-intent keywords (e.g., "write a narrative", "draft a report"). The narrative follows 6 mandatory sections:
1. Signal Description
2. Data Source and Method
3. Statistical Analysis Summary
4. Clinical Assessment
5. Data Limitations
6. Recommendation

All narratives are marked as "DRAFT — For Human Review Only" and are downloadable as `.txt` files.

### 4. Temporal Trend Analysis
Calculates AdverseScore and PRR across **4 rolling quarters** using date-filtered openFDA queries. Classifies trends as:
- **RISING** — AdverseScore increased by 10+ points (emerging signal)
- **STABLE** — Score change within 10 points
- **DECLINING** — Score decreased by 10+ points (signal attenuation)
- **INSUFFICIENT_DATA** — Fewer than 2 valid quarters

Renders a **Plotly dual-axis line chart** (AdverseScore left axis, PRR right axis) with a trend classification badge.

### 5. Comparative Safety Scorecard & Session Memory
**SQLite-backed persistence** automatically saves every completed analysis. Features include:
- **Sidebar History Panel** — Last 20 analyses, sortable by recency, drug name, or score. Click to re-run.
- **Delta Detection** — When a drug is re-queried, automatically surfaces the score change vs the prior analysis: "AdverseScore has increased from 35 to 52 since 2026-01-15."
- **Portfolio Scorecard** — Comparative table showing the latest analysis per drug with columns for AdverseScore, PRR, Confidence, Trend, and Label Status. Rows with AdverseScore > 70 are highlighted.

All persisted data is local to the machine. No cloud storage, no PII, no patient-level data is saved.


## System Architecture

```text
adversescore/
├── app.py                        # Streamlit chat UI with sidebar history & scorecard
├── src/
│   └── adverse_score/
│       ├── client.py             # Core engine: FDA API, scoring math, PRR, label detection, temporal analysis
│       ├── agent_tools.py        # Pydantic schema + @tool function with persistence integration
│       ├── orchestrator.py       # LangGraph agent, system prompt, safety protocols
│       ├── persistence.py        # SQLite persistence layer (AnalysisStore)
│       └── config.py             # Environment & API key validation
├── data/                         # SQLite DB (auto-created, gitignored)
├── docs/                         # Product Requirements Document
├── conftest.py                   # Pytest fixtures
├── test_adversescore.py          # Test suite (150+ tests)
├── requirements.txt              # Dependencies
└── .env                          # API keys (gitignored)
```


## Clinical Guardrails

Unlike standard LLM implementations, AdverseScore enforces multiple layers of safety controls:

- **Diagnosis Lock:** The system is unconditionally forbidden from formulating diagnoses, recommending medication changes, or offering autonomous medical advice. This applies to every response regardless of payload content.
- **Human-in-the-Loop:** High-risk signals (Score > 70 or PRR signal detected) automatically trigger an escalation directive, requiring a qualified clinician to review before any action is taken.
- **Label Penalty System:** Unlabeled adverse events receive amplified weight in the scoring algorithm (up to 2.0x for serious outcomes), ensuring novel safety signals are not underweighted relative to known risks.
- **Narrative Safeguards:** All generated signal narratives are marked "DRAFT — For Human Review Only." Clinical interpretation is encouraged, but definitive diagnoses and causal conclusions are blocked. Interpretive analysis beyond raw data is flagged with: "Note: this interpretation requires clinical validation."
- **Delta Thresholds:** Score changes of 10+ points between analyses are flagged as clinically meaningful, prompting the user to compare against prior analysis context.
- **Demographic Cohorting:** Queries are automatically bracketed into 10-year age cohorts to maintain statistical relevance while preserving sample size.


## Getting Started

### Prerequisites
* Python 3.10+
* openFDA API Key
* OpenAI API Key (GPT-4o)

### Installation
1. Clone the Repository
```bash
git clone https://github.com/your-username/adversescore.git
cd adversescore
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Configure Environment

Create a `.env` file in the root directory:
```
OPENFDA_API_KEY=your_fda_key_here
OPENAI_API_KEY=your_openai_key_here
```

4. Launch the Tool
```bash
streamlit run app.py
```

The `data/` directory for SQLite persistence is created automatically on first run.

### Running Tests
```bash
python -m pytest test_adversescore.py -v
```

---

DISCLAIMER: AdverseScore is a research-grade clinical decision support tool intended for informational purposes only. It is not a medical device and does not constitute medical advice. All outputs require validation by a qualified clinician or pharmacovigilance professional before use in regulatory submissions or clinical decisions.
