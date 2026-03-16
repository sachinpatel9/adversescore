# AdverseScore: Agentic Clinical Safety & Pharmacovigilance Engine

**AdverseScore** is a clinical decision support system that utilizes Agentic AI to transform raw FDA adverse event data into statistically supported safety insights. 
It is specifically designed for clinicians and pharmacovigilance teams in BioPharma. The engine analyzes pharmaceutical risk profiles through the lens of real-world evidence, demographic specificity, and an epidemiological proportional reporting ratio.


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

## 🏗️ System Architecture

```text
adversescore/
├── app.py                  # Streamlit "White-Label" UI
├── src/
│   └── adverse_score/
│       ├── client.py       # Core Statistical Engine (PRR, EPC Discovery, FDA API)
│       ├── agent_tools.py  # Pydantic Tool Definitions for LangChain
│       ├── orchestrator.py # LangGraph Agent & Safety Protocol Logic
│       └── config.py       # Environment & API Validation
├── requirements.txt        # Enterprise-standard dependency tracking
└── .env                    # Protected API Infrastructure
```

## Clinical Guardrails

Unlike standard LLM implementations, AdverseScore enforces a **Diagnosis Lock**. Even if the AI identifies a high-risk signal, it is programmatically forbidden from offering medical advice or treatment changes. It is designed to route findings to an Subject Matter Expert (clinician, PV lead) if risk thresholds are breached.

- **Zero Advice Policy:** The system is programmatically forbidden from formulating diagnoses and medication changes.
- **Human-in-the-Loop:** High-risk signals (Score > 70 or PRSSignalDetected) automatically trigger an escalation directive, requiring an SME to review immediately.
- **Demographic Cohorting:** Queries are automatically bracketed into 10-year age cohorts to maintain statistical relevance while preserving sample size (N)

## Getting Started

### Prerequisites
* Python 3.10+
* openFDA API Key
* OpenAI API Key (GPT-4o)

### Installation
1. Clone the Repository
```bash
git clone [https://github.com/your-username/adversescore.git](https://github.com/your-username/adversescore.git)
cd adversescore
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Configure Environment
- Create a .env file in the root directory:
OPENFDA_API_KEY=your_fda_key_here
OPENAI_API_KEY=your_openai_key_here

4. Launch the Tool 
```bash
streamlit run app.py
```

DISCLAIMER: AdverseScore is a research-grade prototype intended for informational purposes only. It is not a medical device and does not constitute medical advice. This tool is a MVP and is scheduled to be improved on over time.