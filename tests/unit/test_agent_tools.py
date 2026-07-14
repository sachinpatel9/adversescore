"""
Unit tests for agent_tools.py — placeholder module.

The old `get_adverse_score` tool tied to the five-capability composite score
was removed as part of the PSUR consolidation rebuild (Phase 0). This is a
regression guard ensuring the removed symbols do not reappear accidentally.
"""

import adverse_score.agent_tools


class TestAgentToolsPlaceholder:
    """Verifies agent_tools imports cleanly and no longer exposes removed symbols."""

    def test_imports_cleanly(self):
        """adverse_score.agent_tools imports without raising."""
        assert adverse_score.agent_tools is not None

    def test_get_adverse_score_removed(self):
        """Regression guard: get_adverse_score no longer exists on the module."""
        assert not hasattr(adverse_score.agent_tools, "get_adverse_score")

    def test_global_client_removed(self):
        """Regression guard: _global_client no longer exists on the module."""
        assert not hasattr(adverse_score.agent_tools, "_global_client")
