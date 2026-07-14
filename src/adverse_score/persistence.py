import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "adversescore.db"


class AnalysisStore:
    """SQLite-backed persistence for AdverseScore.

    The old single-analysis-per-row schema (composite score, PRR value, peer
    benchmark, trend classification) has been removed as part of the PSUR
    consolidation rebuild (Phase 0, see docs/PSUR_CONSOLIDATION_SCOPE.md).
    The new schema — full consolidated/deduplicated/ranked datasets per
    drug + period, plus conversation history — is built in Phase 7.

    Only the connection lifecycle and context-manager protocol survive here;
    Phase 7 builds its schema on top of this skeleton.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        # No tables yet — schema rebuilt in Phase 7.
        pass

    def __enter__(self) -> "AnalysisStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()
