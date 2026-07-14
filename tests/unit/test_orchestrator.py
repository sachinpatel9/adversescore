"""
Unit tests for orchestrator.py — placeholder module.

The old GPT-4o react-agent wiring was removed as part of the PSUR
consolidation rebuild (Phase 0). `agent_executor` is intentionally `None`
so callers (e.g. app.py) can detect the placeholder state without requiring
OPENAI_API_KEY/OPENFDA_API_KEY just to import this module.
"""


class TestOrchestratorPlaceholder:
    """Verifies the orchestrator placeholder imports cleanly without API keys."""

    def test_import_succeeds_without_api_keys(self, monkeypatch):
        """`from adverse_score.orchestrator import agent_executor` succeeds without raising
        even when OPENAI_API_KEY/OPENFDA_API_KEY are absent from the environment."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENFDA_API_KEY", raising=False)

        # Import already succeeded during test collection (proving no fail-fast at
        # import time); re-importing here is safe and exercises the same module object.
        from adverse_score.orchestrator import agent_executor
        assert agent_executor is None

    def test_agent_executor_is_none(self):
        """agent_executor is the placeholder None value."""
        from adverse_score.orchestrator import agent_executor
        assert agent_executor is None
