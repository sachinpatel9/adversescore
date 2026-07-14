"""
Unit tests for persistence.py — AnalysisStore skeleton.

The old single-analysis-per-row schema (composite score, PRR value, peer
benchmark, trend classification) was removed as part of the PSUR
consolidation rebuild (Phase 0). Only the connection lifecycle and
context-manager protocol survive; Phase 7 builds the new schema on top.
"""

import sqlite3

from adverse_score.persistence import AnalysisStore


class TestPersistenceLayerEmpty:
    """Tests for the AnalysisStore skeleton — construction, no-op schema, context manager."""

    def test_constructs_successfully(self, tmp_path):
        """AnalysisStore() constructs successfully against a tmp_path-backed db."""
        db_path = tmp_path / "empty_test.db"
        store = AnalysisStore(db_path=db_path)
        assert store is not None
        store.close()

    def test_init_schema_does_not_raise(self, tmp_path):
        """_init_schema() is a no-op and does not raise."""
        db_path = tmp_path / "schema_test.db"
        store = AnalysisStore(db_path=db_path)
        store._init_schema()
        store.close()

    def test_no_tables_created(self, tmp_path):
        """Querying sqlite_master for table names returns an empty list — no 'analyses' table exists."""
        db_path = tmp_path / "no_tables_test.db"
        store = AnalysisStore(db_path=db_path)
        cursor = store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        assert tables == []
        store.close()

    def test_context_manager_protocol(self, tmp_path):
        """The context manager protocol still works: `with AnalysisStore(...) as store:`."""
        db_path = tmp_path / "cm_test.db"
        with AnalysisStore(db_path=db_path) as store:
            assert store is not None
            cursor = store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            assert cursor.fetchall() == []
        # Connection should be closed after exiting the context manager
        try:
            store._conn.execute("SELECT 1")
            closed = False
        except sqlite3.ProgrammingError:
            closed = True
        assert closed, "Connection should be closed after exiting the context manager"
