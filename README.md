# AdverseScore: Agentic Pharmacovigilance Engine

**AdverseScore** is a clinical decision support tool that utilizes Agentic AI to analyze FDA adverse event data for pharmaceuticals. It implements a proprietary scoring logic that weighs severity, recency, and label-awareness to provide a safety risk profile (0-100) for target medications.



## Architecture Overview

The system is built on a four-module "Safety-First" architecture:

1.  **Ingestion & Discovery:** Robust integration with the openFDA API featuring automated retry logic and session persistence.
2.  **The Scoring Engine:** Calculates weighted risk scores based on event severity (e.g., Death, Hospitalization) and recency decay.
3.  **Grounding & Validation:** Implements a **Confidence Index** based on sample size and data integrity, ensuring the AI knows when data is sparse.
4.  **Agentic Guardrails:** Deterministic Python-level booleans (`diagnosis_lock`, `requires_human_review`) that control LLM behavior.

## Clinical Guardrails

Unlike standard LLM implementations, AdverseScore enforces a **Diagnosis Lock**. Even if the AI identifies a high-risk signal, it is programmatically forbidden from offering medical advice or treatment changes. It is designed to route findings to a human Clinical Safety Officer if risk thresholds are breached.

## Getting Started

### Prerequisites
* Python 3.10+
* openFDA API Key
* OpenAI API Key

### Installation
```bash
pip install langchain langchain-openai langgraph requests python-dotenv