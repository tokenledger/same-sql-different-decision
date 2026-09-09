"""SQLite execution harness and execution-match correctness labeling."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

from .config import db_path

TIMEOUT_SECONDS = 30.0
# Checked every N virtual-machine instructions by the progress handler.
_PROGRESS_OPS = 1000


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] | None
    error: str | None

    @property
    def status(self) -> str:
        if self.ok:
            return "ok"
        return "timeout" if self.error == "timeout" else "error"


def execute(db_id: str, sql: str, timeout: float = TIMEOUT_SECONDS) -> ExecResult:
    """Run `sql` read-only against a Spider database, aborting on timeout."""
    deadline = time.monotonic() + timeout
    timed_out = False

    def _abort() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1  # non-zero aborts the running statement
        return 0

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
        # Spider databases contain text stored under inconsistent encodings.
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        conn.set_progress_handler(_abort, _PROGRESS_OPS)
        rows = conn.execute(sql).fetchall()
        return ExecResult(ok=True, rows=rows, error=None)
    except Exception as e:  # noqa: BLE001 - any failure is just an incorrect query
        return ExecResult(ok=False, rows=None, error="timeout" if timed_out else str(e))
    finally:
        if conn is not None:
            conn.close()


def _normalize_cell(v):
    """Make cells comparable across float noise and int/float mismatches."""
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def _normalize_rows(rows: list[tuple], ordered: bool) -> list:
    norm = [tuple(_normalize_cell(c) for c in row) for row in rows]
    if ordered:
        return norm
    # Order-insensitive multiset comparison; str() keys make mixed types sortable.
    return sorted(norm, key=lambda r: tuple(str(c) for c in r))


def has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE) is not None


def result_match(db_id: str, pred_sql: str, gold_sql: str) -> tuple[bool, str]:
    """Execution-match label: does `pred_sql` return the gold result set?

    Row order is only enforced when the gold query has an ORDER BY, matching
    Spider's execution-accuracy convention.
    """
    pred = execute(db_id, pred_sql)
    if not pred.ok:
        return False, pred.status

    gold = execute(db_id, gold_sql)
    if not gold.ok:
        # A gold query that will not run means the item is unusable, not that
        # the prediction is wrong.
        return False, f"gold_{gold.status}"

    ordered = has_order_by(gold_sql)
    match = _normalize_rows(pred.rows, ordered) == _normalize_rows(gold.rows, ordered)
    return match, "ok"
