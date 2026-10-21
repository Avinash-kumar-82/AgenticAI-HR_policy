import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def _connect() -> sqlite3.Connection:
    db_path = Path(settings.audit_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=5)


def init_db() -> None:
    with closing(_connect()) as con, con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                source_used TEXT NOT NULL,
                trace_json TEXT NOT NULL
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_query_audit_created_at "
            "ON query_audit (created_at)"
        )


def write_audit(
    question: str,
    source_used: str,
    trace: list[str],
) -> None:
    """Record a query. Never raises: auditing must not break the answer."""
    try:
        init_db()  # cheap (IF NOT EXISTS); guarantees the table exists

        with closing(_connect()) as con, con:
            con.execute(
                """
                INSERT INTO query_audit (
                    created_at, question, source_used, trace_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    question,
                    source_used,
                    json.dumps(trace),
                ),
            )
    except Exception:
        logger.exception("Audit write failed")