import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DB_DIR / "adversescore.db"


class AnalysisStore:
    """SQLite-backed persistence for completed AdverseScore analyses.

    Stores aggregate statistical outputs only — no PII, no patient data.
    All data is local to the machine.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                adverse_score REAL,
                prr_value REAL,
                peer_benchmark_avg REAL,
                confidence_level TEXT,
                trend_classification TEXT,
                label_status TEXT,
                report_count INTEGER,
                signal_narrative TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_drug_name ON analyses(drug_name);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON analyses(timestamp DESC);
        """)
        self._conn.commit()

    def save_analysis(self, payload: dict) -> int:
        """Extract fields from an agent_payload dict and INSERT a row. Returns the row id."""
        clinical = payload["clinical_signal"]
        metadata = payload["metadata"]
        integrity = payload["data_integrity"]

        # PRR: pharmacovigilance_metrics can be None
        pv_metrics = payload.get("pharmacovigilance_metrics")
        prr_value = pv_metrics.get("prr") if pv_metrics else None

        # Temporal: optional
        temporal = payload.get("temporal_analysis")
        trend = temporal.get("trend_classification") if temporal else None

        cursor = self._conn.cursor()
        cursor.execute(
            """INSERT INTO analyses
               (drug_name, timestamp, adverse_score, prr_value, peer_benchmark_avg,
                confidence_level, trend_classification, label_status, report_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                clinical["drug_target"],
                metadata["timestamp"],
                clinical["adverse_score"],
                prr_value,
                clinical.get("class_benchmark_avg"),
                integrity["confidence_level"],
                trend,
                clinical.get("label_status"),
                integrity["report_count"],
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent analyses by insertion order (id DESC)."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_prior_analysis(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Return the most recent saved analysis for a given drug (case-insensitive)."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM analyses WHERE UPPER(drug_name) = UPPER(?) ORDER BY timestamp DESC LIMIT 1",
            (drug_name,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_portfolio(self) -> List[Dict[str, Any]]:
        """Return the latest analysis per drug for the comparative scorecard."""
        cursor = self._conn.cursor()
        cursor.execute(
            """SELECT * FROM analyses
               WHERE id IN (SELECT MAX(id) FROM analyses GROUP BY UPPER(drug_name))
               ORDER BY adverse_score DESC"""
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_narrative(self, drug_name: str, narrative: str) -> None:
        """Save a signal narrative to the most recent analysis row for a drug."""
        cursor = self._conn.cursor()
        cursor.execute(
            """UPDATE analyses SET signal_narrative = ?
               WHERE id = (
                   SELECT id FROM analyses
                   WHERE UPPER(drug_name) = UPPER(?)
                   ORDER BY timestamp DESC LIMIT 1
               )""",
            (narrative, drug_name),
        )
        self._conn.commit()

    def __enter__(self) -> "AnalysisStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()
