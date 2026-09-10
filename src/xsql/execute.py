"""SQLite execution harness and execution-match correctness labeling."""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from itertools import permutations

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


def _sort_key(v):
    """Type-aware sort key so that 1 and 1.0 (equal) sort identically.

    A str() key would place 1 and 1.0 differently and make two equal
    multisets compare unequal after sorting."""
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, (int, float)):
        return (1, float(v), "")
    return (2, 0.0, str(v))


def _normalize_rows(rows: list[tuple], ordered: bool) -> list:
    norm = [tuple(_normalize_cell(c) for c in row) for row in rows]
    if ordered:
        return norm
    # Order-insensitive multiset comparison.
    return sorted(norm, key=lambda r: tuple(_sort_key(c) for c in r))


def has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE) is not None


# Column permutations are enumerated exhaustively up to this width. Wider
# results (six gold queries in the corpus, 7 to 13 columns) use a
# column-matching search instead, which is exact and avoids factorial blowup.
MAX_PERMUTED_COLUMNS = 6


def _column_permutations(rows: list[tuple], width: int):
    """Yield `rows` under every reordering of its columns (identity first)."""
    for perm in permutations(range(width)):
        yield [tuple(r[i] for i in perm) for r in rows]


def _columns(rows: list[tuple], width: int) -> list[tuple]:
    return [tuple(r[i] for r in rows) for i in range(width)]


def _column_key(col: tuple, ordered: bool) -> tuple:
    """Column contents as a sequence (ordered) or a sorted multiset (unordered)."""
    return col if ordered else tuple(sorted(col, key=_sort_key))


def _wide_match(pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool) -> bool:
    """Exact column-permutation match for wide results without factorial blowup.

    Cells are normalized first (same as the narrow path). Each gold column is
    matched to a distinct predicted column with the same contents, compared
    as a sequence when row order matters and as a multiset otherwise. Because
    a column-wise match is necessary but not sufficient (rows could be
    recombined across columns), every candidate assignment is verified with
    the whole-row comparison."""
    width = len(gold_rows[0])
    pred_norm = [tuple(_normalize_cell(c) for c in r) for r in pred_rows]
    gold_norm_rows = [tuple(_normalize_cell(c) for c in r) for r in gold_rows]
    pred_cols = [_column_key(c, ordered) for c in _columns(pred_norm, width)]
    gold_cols = [_column_key(c, ordered) for c in _columns(gold_norm_rows, width)]
    cands = [[j for j, pc in enumerate(pred_cols) if pc == gc] for gc in gold_cols]
    gold_norm = _normalize_rows(gold_rows, ordered)
    for perm in _assignments(cands, width):
        permuted = [tuple(r[i] for i in perm) for r in pred_rows]
        if _normalize_rows(permuted, ordered) == gold_norm:
            return True
    return False


def _assignments(cands: list[list[int]], width: int):
    """All injective assignments gold column -> predicted column."""
    used: set[int] = set()
    perm: list[int] = []

    def rec(i: int):
        if i == width:
            yield list(perm)
            return
        for j in cands[i]:
            if j in used:
                continue
            used.add(j)
            perm.append(j)
            yield from rec(i + 1)
            perm.pop()
            used.discard(j)

    yield from rec(0)


def rows_match(pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool) -> bool:
    """Compare result sets up to column order.

    Spider's evaluator maps returned columns onto the parsed SELECT list rather
    than comparing them positionally, so `SELECT max(x), min(x)` and
    `SELECT min(x), max(x)` are the same answer. A positional comparison marks
    the second one wrong. We follow Spider: the prediction matches if some
    permutation of its columns equals the gold result (rows as a multiset,
    or in order when the gold query has an ORDER BY).
    """
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True
    width = len(gold_rows[0])
    if any(len(r) != width for r in pred_rows):
        return False
    if width > MAX_PERMUTED_COLUMNS:
        return _wide_match(pred_rows, gold_rows, ordered)
    gold_norm = _normalize_rows(gold_rows, ordered)
    for permuted in _column_permutations(pred_rows, width):
        if _normalize_rows(permuted, ordered) == gold_norm:
            return True
    return False


def result_match(db_id: str, pred_sql: str, gold_sql: str) -> tuple[bool, str]:
    """Execution-match label: does `pred_sql` return the gold result set?

    Row order is only enforced when the gold query has an ORDER BY, and column
    order is never enforced, matching Spider's execution-accuracy convention.
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
    return rows_match(pred.rows, gold.rows, ordered), "ok"
