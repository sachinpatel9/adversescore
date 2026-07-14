"""Agent orchestration — placeholder.

The old single GPT-4o react-agent wired to the five-capability composite-score
tool has been removed as part of the PSUR consolidation rebuild (Phase 0, see
docs/PSUR_CONSOLIDATION_SCOPE.md). The new system prompt, guardrails, and tool
wiring for the PSUR consolidation job are built in Phase 8, on top of the
Phase 1-7 pipeline modules.

`agent_executor` is intentionally `None` here so callers (e.g. app.py) can
detect the placeholder state without requiring OPENAI_API_KEY/OPENFDA_API_KEY
just to import this module.
"""

agent_executor = None
