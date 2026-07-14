"""
Unit tests for persistence.py — ConsolidationStore (Phase 7).

Covers schema creation, ConsolidationResult round-trip serialization
(including the documented chunk-report trim), multi-row/no-overwrite save
semantics, latest-consolidation lookup, lightweight listing, and
conversation-history storage. Pure local SQLite — no FDA API, no network.
"""
import dataclasses
import sqlite3
from datetime import date

import pytest

from adverse_score.consolidation import ConsolidationResult
from adverse_score.deduplication import DedupResult
from adverse_score.drug_identity import DrugIdentity
from adverse_score.fda_client import ChunkResult, PSURRetrievalResult
from adverse_score.label_classifier import LabelClassificationResult
from adverse_score.persistence import ConsolidationStore
from adverse_score.ranking import RankedSignal, RankingResult


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_drug_identity(auth_date=date(2014, 9, 4)):
    return DrugIdentity(
        canonical_name="KEYTRUDA",
        brand_names=["KEYTRUDA"],
        generic_names=["PEMBROLIZUMAB"],
        substance_names=["PEMBROLIZUMAB"],
        application_numbers=["BLA125514"],
        market_authorization_date=auth_date,
        market_authorization_date_source="drugsfda.json:BLA125514",
        resolution_confidence="EXACT",
    )


def _make_retrieval():
    sample_reports = [
        {
            "report_id": "RPT-001",
            "date": "20240101",
            "severity": "Serious",
            "is_death": False,
            "is_hospitalization": True,
            "symptoms": "NAUSEA",
            "symptom_list": ["NAUSEA"],
            "drug_names": ["KEYTRUDA"],
            "safetyreportversion": 1,
            "reactions": [{"term": "NAUSEA", "outcome_code": 1}],
            "company": "PHARMA-001",
        },
        {
            "report_id": "RPT-002",
            "date": "20240102",
            "severity": "Serious",
            "is_death": True,
            "is_hospitalization": False,
            "symptoms": "CARDIAC ARREST",
            "symptom_list": ["CARDIAC ARREST"],
            "drug_names": ["KEYTRUDA"],
            "safetyreportversion": 1,
            "reactions": [{"term": "CARDIAC ARREST", "outcome_code": 5}],
            "company": "PHARMA-002",
        },
    ]
    chunk = ChunkResult(
        label="chunk-0-20240101-to-20240331",
        start_date="20240101",
        end_date="20240331",
        reports=list(sample_reports),
        retrieved_count=2,
        estimated_total_count=2,
        truncated=False,
        error=None,
    )
    return PSURRetrievalResult(
        canonical_query_variants=["KEYTRUDA", "PEMBROLIZUMAB"],
        period_start="20240101",
        period_end="20240630",
        chunks=[chunk],
        reports=list(sample_reports),
        total_retrieved_count=2,
        total_estimated_count=2,
        any_chunk_truncated=False,
        used_fallback_anchor=False,
        fallback_reason=None,
    )


def _make_dedup(retrieval):
    return DedupResult(
        reports=list(retrieval.reports),
        total_input_count=2,
        total_output_count=2,
        removed_by_exact_id=0,
        removed_by_heuristic=0,
        audit_trail=[],
    )


def _make_ranking(ranked_signals=None):
    if ranked_signals is None:
        ranked_signals = [
            RankedSignal(
                symptom="CARDIAC ARREST",
                rank=1,
                seriousness_tier="DEATH",
                strength_of_evidence_tier="UNKNOWN_LABEL_STATUS",
                reversibility_tier="FATAL",
                public_health_tier="LOW",
                prr_metrics={
                    "prr": 0.0,
                    "ci_lower": 0.0,
                    "signal_detected": False,
                    "target_symptom": "CARDIAC ARREST",
                    "drug_cases": 1,
                    "class_cases": 0,
                    "label_status": "LABEL_STATUS_UNKNOWN",
                },
                report_count=1,
            ),
            RankedSignal(
                symptom="NAUSEA",
                rank=2,
                seriousness_tier="HOSPITALIZATION",
                strength_of_evidence_tier="UNKNOWN_LABEL_STATUS",
                reversibility_tier="REVERSIBLE",
                public_health_tier="LOW",
                prr_metrics={
                    "prr": 0.0,
                    "ci_lower": 0.0,
                    "signal_detected": False,
                    "target_symptom": "NAUSEA",
                    "drug_cases": 1,
                    "class_cases": 0,
                    "label_status": "LABEL_STATUS_UNKNOWN",
                },
                report_count=1,
            ),
        ]
    label_summary = LabelClassificationResult(
        statuses={"CARDIAC ARREST": "LABEL_STATUS_UNKNOWN", "NAUSEA": "LABEL_STATUS_UNKNOWN"},
        total_symptoms=2,
        labeled_count=0,
        unlabeled_count=0,
        unknown_count=2,
    )
    return RankingResult(
        ranked_signals=ranked_signals,
        label_summary=label_summary,
        formula_version="1.0",
        total_signals=len(ranked_signals),
    )


def _make_consolidation_result(canonical_name="KEYTRUDA", period="6mo",
                                auth_date=date(2014, 9, 4), ranked_signals=None):
    identity = _make_drug_identity(auth_date=auth_date)
    identity = dataclasses.replace(identity, canonical_name=canonical_name)
    retrieval = _make_retrieval()
    dedup = _make_dedup(retrieval)
    ranking = _make_ranking(ranked_signals=ranked_signals)
    return ConsolidationResult(
        drug_identity=identity,
        period=period,
        retrieval=retrieval,
        dedup=dedup,
        ranking=ranking,
        pharm_class="PD-1/PD-L1 INHIBITOR",
        class_counts_available=True,
        formula_version="1.0",
    )


# ── Construction / Context Manager ─────────────────────────────────────────

class TestConsolidationStoreLifecycle:
    def test_constructs_successfully(self, tmp_path):
        db_path = tmp_path / "empty_test.db"
        store = ConsolidationStore(db_path=db_path)
        assert store is not None
        store.close()

    def test_init_schema_does_not_raise(self, tmp_path):
        db_path = tmp_path / "schema_test.db"
        store = ConsolidationStore(db_path=db_path)
        store._init_schema()
        store.close()

    def test_context_manager_protocol(self, tmp_path):
        db_path = tmp_path / "cm_test.db"
        with ConsolidationStore(db_path=db_path) as store:
            assert store is not None
        try:
            store._conn.execute("SELECT 1")
            closed = False
        except sqlite3.ProgrammingError:
            closed = True
        assert closed, "Connection should be closed after exiting the context manager"


# ── Schema Creation ──────────────────────────────────────────────────────

class TestSchemaCreation:
    def test_tables_exist(self, temp_store):
        cursor = temp_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "consolidations" in tables
        assert "conversation_messages" in tables

    def test_consolidations_columns(self, temp_store):
        cursor = temp_store._conn.execute("PRAGMA table_info(consolidations)")
        columns = {row["name"] for row in cursor.fetchall()}
        assert columns == {"id", "canonical_name", "period", "created_at", "result_json"}

    def test_conversation_messages_columns(self, temp_store):
        cursor = temp_store._conn.execute("PRAGMA table_info(conversation_messages)")
        columns = {row["name"] for row in cursor.fetchall()}
        assert columns == {
            "id", "session_id", "consolidation_id", "role", "content", "created_at",
        }


# ── Round-Trip Fidelity ────────────────────────────────────────────────────

class TestRoundTripFidelity:
    def test_full_round_trip(self, temp_store):
        original = _make_consolidation_result()
        new_id = temp_store.save_consolidation(original)
        loaded = temp_store.load_consolidation(new_id)

        assert loaded is not None
        # Chunk-level reports are trimmed on save — documented, intentional.
        assert loaded.retrieval.chunks[0].reports == []
        assert original.retrieval.chunks[0].reports != []

        # Full report data survives at the retrieval/dedup level (NOT trimmed).
        assert loaded.retrieval.reports == original.retrieval.reports
        assert loaded.dedup.reports == original.dedup.reports

        # Everything else must be a byte-for-byte structural match once the
        # known chunk-report trim is accounted for — rebuild an "expected"
        # copy with that field zeroed out for a full equality check.
        expected_chunks = [
            dataclasses.replace(c, reports=[]) for c in original.retrieval.chunks
        ]
        expected_retrieval = dataclasses.replace(original.retrieval, chunks=expected_chunks)
        expected = dataclasses.replace(original, retrieval=expected_retrieval)
        assert loaded == expected

    def test_full_round_trip_trims_all_chunks_not_just_first(self, temp_store):
        """Regression guard for the chunk-report trim: the base fixture used by
        test_full_round_trip only has one chunk, which would not catch a bug
        that trimmed chunks[0] only (e.g. an off-by-one or hardcoded index)
        instead of looping over every chunk. This test builds a
        PSURRetrievalResult with two chunks and asserts BOTH are trimmed on
        save, while retrieval.reports and dedup.reports (holding the union of
        both chunks' reports) remain fully intact."""
        chunk_a_reports = [
            {
                "report_id": "RPT-A1",
                "date": "20240101",
                "severity": "Serious",
                "is_death": False,
                "is_hospitalization": True,
                "symptoms": "NAUSEA",
                "symptom_list": ["NAUSEA"],
                "drug_names": ["KEYTRUDA"],
                "safetyreportversion": 1,
                "reactions": [{"term": "NAUSEA", "outcome_code": 1}],
                "company": "PHARMA-A1",
            },
        ]
        chunk_b_reports = [
            {
                "report_id": "RPT-B1",
                "date": "20240401",
                "severity": "Serious",
                "is_death": True,
                "is_hospitalization": False,
                "symptoms": "CARDIAC ARREST",
                "symptom_list": ["CARDIAC ARREST"],
                "drug_names": ["KEYTRUDA"],
                "safetyreportversion": 1,
                "reactions": [{"term": "CARDIAC ARREST", "outcome_code": 5}],
                "company": "PHARMA-B1",
            },
        ]
        chunk_a = ChunkResult(
            label="chunk-0-20240101-to-20240331",
            start_date="20240101",
            end_date="20240331",
            reports=list(chunk_a_reports),
            retrieved_count=1,
            estimated_total_count=1,
            truncated=False,
            error=None,
        )
        chunk_b = ChunkResult(
            label="chunk-1-20240401-to-20240630",
            start_date="20240401",
            end_date="20240630",
            reports=list(chunk_b_reports),
            retrieved_count=1,
            estimated_total_count=1,
            truncated=False,
            error=None,
        )
        merged_reports = chunk_a_reports + chunk_b_reports
        retrieval = PSURRetrievalResult(
            canonical_query_variants=["KEYTRUDA", "PEMBROLIZUMAB"],
            period_start="20240101",
            period_end="20240630",
            chunks=[chunk_a, chunk_b],
            reports=list(merged_reports),
            total_retrieved_count=2,
            total_estimated_count=2,
            any_chunk_truncated=False,
            used_fallback_anchor=False,
            fallback_reason=None,
        )
        dedup = _make_dedup(retrieval)
        ranking = _make_ranking()
        original = ConsolidationResult(
            drug_identity=_make_drug_identity(),
            period="6mo",
            retrieval=retrieval,
            dedup=dedup,
            ranking=ranking,
            pharm_class="PD-1/PD-L1 INHIBITOR",
            class_counts_available=True,
            formula_version="1.0",
        )

        new_id = temp_store.save_consolidation(original)
        loaded = temp_store.load_consolidation(new_id)

        assert loaded is not None
        assert len(loaded.retrieval.chunks) == 2
        # Every chunk must be trimmed, not just the first.
        assert loaded.retrieval.chunks[0].reports == []
        assert loaded.retrieval.chunks[1].reports == []
        # Originals (in-memory, pre-save) must remain untouched.
        assert original.retrieval.chunks[0].reports != []
        assert original.retrieval.chunks[1].reports != []
        # Full merged report data (both chunks' reports) survives at the
        # retrieval/dedup level.
        assert loaded.retrieval.reports == merged_reports
        assert loaded.dedup.reports == merged_reports

    def test_market_authorization_date_none_round_trip(self, temp_store):
        """OTC-drug case: market_authorization_date=None must survive without crashing."""
        original = _make_consolidation_result(auth_date=None)
        new_id = temp_store.save_consolidation(original)
        loaded = temp_store.load_consolidation(new_id)

        assert loaded is not None
        assert loaded.drug_identity.market_authorization_date is None

    def test_empty_ranked_signals_round_trip(self, temp_store):
        original = _make_consolidation_result(ranked_signals=[])
        assert original.ranking.total_signals == 0
        new_id = temp_store.save_consolidation(original)
        loaded = temp_store.load_consolidation(new_id)

        assert loaded is not None
        assert loaded.ranking.ranked_signals == []
        assert loaded.ranking.total_signals == 0

    def test_load_consolidation_returns_none_when_missing(self, temp_store):
        assert temp_store.load_consolidation(99999) is None


# ── Multiple Saves / No Overwrite ─────────────────────────────────────────

class TestMultipleSaves:
    def test_multiple_saves_do_not_overwrite(self, temp_store):
        first = _make_consolidation_result(canonical_name="KEYTRUDA", period="6mo")
        second = _make_consolidation_result(canonical_name="KEYTRUDA", period="6mo")

        id1 = temp_store.save_consolidation(first)
        id2 = temp_store.save_consolidation(second)

        assert id1 != id2
        loaded1 = temp_store.load_consolidation(id1)
        loaded2 = temp_store.load_consolidation(id2)
        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.drug_identity.canonical_name == "KEYTRUDA"
        assert loaded2.drug_identity.canonical_name == "KEYTRUDA"


# ── get_latest_consolidation ────────────────────────────────────────────────

class TestGetLatestConsolidation:
    def test_returns_most_recently_inserted(self, temp_store):
        older = _make_consolidation_result(canonical_name="OPDIVO", period="1yr")
        newer = _make_consolidation_result(canonical_name="OPDIVO", period="1yr")

        temp_store.save_consolidation(older)
        newer_id = temp_store.save_consolidation(newer)

        latest = temp_store.get_latest_consolidation("OPDIVO", "1yr")
        assert latest is not None
        # Confirm it's the row we most recently inserted, by cross-checking
        # against a direct load of the known-newer id.
        assert latest == temp_store.load_consolidation(newer_id)

    def test_returns_none_when_no_match(self, temp_store):
        assert temp_store.get_latest_consolidation("NONEXISTENT_DRUG", "6mo") is None


# ── list_consolidations ─────────────────────────────────────────────────────

class TestListConsolidations:
    def test_metadata_only_shape(self, temp_store):
        result = _make_consolidation_result(canonical_name="HUMIRA", period="2yr")
        new_id = temp_store.save_consolidation(result)

        listing = temp_store.list_consolidations()
        assert len(listing) == 1
        row = listing[0]
        assert set(row.keys()) == {"id", "canonical_name", "period", "created_at"}
        assert row["id"] == new_id
        assert row["canonical_name"] == "HUMIRA"
        assert row["period"] == "2yr"

    def test_filter_by_canonical_name(self, temp_store):
        temp_store.save_consolidation(
            _make_consolidation_result(canonical_name="HUMIRA", period="2yr"))
        temp_store.save_consolidation(
            _make_consolidation_result(canonical_name="ENBREL", period="2yr"))

        all_rows = temp_store.list_consolidations()
        assert len(all_rows) == 2

        humira_rows = temp_store.list_consolidations(canonical_name="HUMIRA")
        assert len(humira_rows) == 1
        assert humira_rows[0]["canonical_name"] == "HUMIRA"


# ── Conversation History ────────────────────────────────────────────────────

class TestConversationHistory:
    def test_save_and_retrieve_interleaved_sessions(self, temp_store):
        result = _make_consolidation_result()
        consolidation_id = temp_store.save_consolidation(result)

        temp_store.save_message("session-A", "user", "Hello A1")
        temp_store.save_message("session-B", "user", "Hello B1")
        temp_store.save_message(
            "session-A", "assistant", "Reply A2", consolidation_id=consolidation_id)
        temp_store.save_message("session-B", "assistant", "Reply B2")

        history_a = temp_store.get_conversation_history("session-A")
        history_b = temp_store.get_conversation_history("session-B")

        assert [m["content"] for m in history_a] == ["Hello A1", "Reply A2"]
        assert [m["content"] for m in history_b] == ["Hello B1", "Reply B2"]

        # consolidation_id round-trips correctly for both the None and
        # populated case.
        assert history_a[0]["consolidation_id"] is None
        assert history_a[1]["consolidation_id"] == consolidation_id

    def test_get_conversation_history_empty_session(self, temp_store):
        assert temp_store.get_conversation_history("no-such-session") == []
