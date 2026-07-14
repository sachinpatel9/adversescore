"""SQLite-backed persistence for AdverseScore — Phase 7 of the PSUR rebuild.

The old single-analysis-per-row schema (composite score, PRR value, peer
benchmark, trend classification) was removed as part of the PSUR
consolidation rebuild (Phase 0, see docs/PSUR_CONSOLIDATION_SCOPE.md). This
module rebuilds persistence around the Phase 6 `ConsolidationResult` shape:
one row per consolidated/deduplicated/ranked dataset (drug + period), plus a
separate table for multi-turn conversation history (Phase 8 will define the
role vocabulary; this layer stores whatever role string it's given).

Serialization strategy: `dataclasses.asdict()` recursively converts the
entire `ConsolidationResult` tree (including every nested Phase 1-6
dataclass) into plain dicts, which `json.dumps` can already handle. Only two
values need special handling: `date` objects (not JSON-serializable) and a
deliberate storage-size trim of `retrieval.chunks[*].reports` (redundant
per-chunk report data already folded into `retrieval.reports` and, after
dedup, `dedup.reports`).

This module intentionally imports every Phase 1-6 dataclass it serializes —
unlike pure math/logic modules (`ranking.py`, `deduplication.py`, etc.) that
avoid importing `FDAClient` to stay HTTP-free, persistence's entire job is
translating those dataclasses to/from storage, so the imports are the point,
not a layering violation. No HTTP calls are made anywhere in this module.
"""
import dataclasses
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from .consolidation import ConsolidationResult
from .deduplication import DedupResult
from .drug_identity import DrugIdentity
from .fda_client import ChunkResult, PSURRetrievalResult
from .label_classifier import LabelClassificationResult
from .logger import get_logger, log_event
from .ranking import RankedSignal, RankingResult

logger = get_logger("persistence")

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "adversescore.db"


# ── Serialization ──────────────────────────────────────────────────────────

def _serialize_consolidation_result(result: ConsolidationResult) -> str:
    """Converts a ConsolidationResult to a JSON string for storage.

    Uses dataclasses.asdict() to recursively flatten the entire nested
    dataclass tree (plain dicts, e.g. flattened report dicts, pass through
    unchanged). Only post-processes what asdict() can't handle on its own:
    the date -> isoformat conversion, and the LOCKED DECISION to trim
    retrieval.chunks[*].reports (a strict subset of data already present in
    retrieval.reports and dedup.reports — do NOT trim those two).
    """
    raw = dataclasses.asdict(result)

    mad = raw["drug_identity"]["market_authorization_date"]
    raw["drug_identity"]["market_authorization_date"] = mad.isoformat() if mad else None

    for chunk in raw["retrieval"]["chunks"]:
        chunk["reports"] = []

    return json.dumps(raw)


def _deserialize_consolidation_result(data: dict) -> ConsolidationResult:
    """Reconstructs a ConsolidationResult from its JSON-decoded dict form.

    Bottom-up reconstruction: rebuild each nested dataclass from the leaves
    inward, then compose the top-level ConsolidationResult. Note that
    retrieval.chunks[*].reports will correctly be [] after reload even if the
    original had data there — that's the intended, documented trim from
    _serialize_consolidation_result, not a bug.
    """
    identity_data = dict(data["drug_identity"])
    mad = identity_data["market_authorization_date"]
    identity_data["market_authorization_date"] = date.fromisoformat(mad) if mad else None
    drug_identity = DrugIdentity(**identity_data)

    chunks = [ChunkResult(**c) for c in data["retrieval"]["chunks"]]
    retrieval = PSURRetrievalResult(**{**data["retrieval"], "chunks": chunks})

    dedup = DedupResult(**data["dedup"])

    label_summary = LabelClassificationResult(**data["ranking"]["label_summary"])
    ranked_signals = [RankedSignal(**s) for s in data["ranking"]["ranked_signals"]]
    ranking = RankingResult(
        ranked_signals=ranked_signals,
        label_summary=label_summary,
        formula_version=data["ranking"]["formula_version"],
        total_signals=data["ranking"]["total_signals"],
    )

    return ConsolidationResult(
        drug_identity=drug_identity,
        period=data["period"],
        retrieval=retrieval,
        dedup=dedup,
        ranking=ranking,
        pharm_class=data["pharm_class"],
        class_counts_available=data["class_counts_available"],
        formula_version=data["formula_version"],
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsolidationStore:
    """SQLite-backed persistence for consolidated PSUR datasets and
    conversation history.

    Schema (created in `_init_schema`):
    - `consolidations`: one row per `consolidate_psur()` result, keyed by
      `canonical_name` + `period`, with the full serialized result as JSON.
      Never upserted — every save inserts a new row, preserving history.
    - `conversation_messages`: multi-turn chat history, optionally linked to
      a `consolidations` row via nullable `consolidation_id`.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS consolidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                period TEXT NOT NULL,
                created_at TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_consolidations_lookup
                ON consolidations(canonical_name, period, created_at DESC)
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                consolidation_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (consolidation_id) REFERENCES consolidations(id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_session
                ON conversation_messages(session_id, id)
            """
        )
        self._conn.commit()

    # ── Consolidations ─────────────────────────────────────────────────

    def save_consolidation(self, result: ConsolidationResult) -> int:
        """Serializes and inserts a new row (never upserts/overwrites).
        Returns the new row's id."""
        canonical_name = result.drug_identity.canonical_name
        period = result.period
        created_at = _utc_now_iso()
        result_json = _serialize_consolidation_result(result)

        cursor = self._conn.execute(
            """
            INSERT INTO consolidations (canonical_name, period, created_at, result_json)
            VALUES (?, ?, ?, ?)
            """,
            (canonical_name, period, created_at, result_json),
        )
        self._conn.commit()
        new_id = cursor.lastrowid
        log_event(logger, "consolidation_saved", id=new_id,
                  canonical_name=canonical_name, period=period)
        return new_id

    def load_consolidation(self, consolidation_id: int) -> Optional[ConsolidationResult]:
        """Loads and reconstructs by id. Returns None if not found."""
        row = self._conn.execute(
            "SELECT result_json FROM consolidations WHERE id = ?",
            (consolidation_id,),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row["result_json"])
        return _deserialize_consolidation_result(data)

    def get_latest_consolidation(self, canonical_name: str, period: str) -> Optional[ConsolidationResult]:
        """Most recent row for a canonical_name + period, reconstructed.
        None if none exist."""
        row = self._conn.execute(
            """
            SELECT result_json FROM consolidations
            WHERE canonical_name = ? AND period = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (canonical_name, period),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row["result_json"])
        return _deserialize_consolidation_result(data)

    def list_consolidations(self, canonical_name: Optional[str] = None) -> list:
        """Lightweight metadata only (id, canonical_name, period, created_at)
        — does not decode result_json. Optionally filtered by canonical_name."""
        if canonical_name is None:
            rows = self._conn.execute(
                """
                SELECT id, canonical_name, period, created_at FROM consolidations
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, canonical_name, period, created_at FROM consolidations
                WHERE canonical_name = ?
                ORDER BY created_at DESC, id DESC
                """,
                (canonical_name,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Conversation History ───────────────────────────────────────────

    def save_message(self, session_id: str, role: str, content: str,
                      consolidation_id: Optional[int] = None) -> int:
        """Inserts one conversation message. Returns its id."""
        created_at = _utc_now_iso()
        cursor = self._conn.execute(
            """
            INSERT INTO conversation_messages
                (session_id, consolidation_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, consolidation_id, role, content, created_at),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_conversation_history(self, session_id: str) -> list:
        """All messages for a session_id, ordered by insertion (id ASC).
        Returned as a list of dicts."""
        rows = self._conn.execute(
            """
            SELECT id, session_id, consolidation_id, role, content, created_at
            FROM conversation_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ── Connection Lifecycle ───────────────────────────────────────────

    def __enter__(self) -> "ConsolidationStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()
