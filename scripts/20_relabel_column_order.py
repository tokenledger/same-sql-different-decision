"""Relabel candidates with a column-order-insensitive execution comparator.

The original labels compared result columns positionally, so a candidate that
returned the requested columns in a different order than the gold query was
marked incorrect. Spider's evaluator does not enforce column order. This script
re-executes every candidate and gold query (SQLite, no model inference),
rewrites the `correct` field in candidates.jsonl and every scores_*.jsonl that
scores these candidates (data/big, data/mitigation, data/architecture), and
records exactly which labels changed in results/relabel_column_order.md.

Run:  XSQL_RUN=big python scripts/20_relabel_column_order.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsql.config import DATA_DIR, RESULTS_DIR, SLICE_DIR  # noqa: E402

# Score files outside the run directory that carry the same candidate labels.
DEPENDENT_DIRS = [DATA_DIR / "mitigation", DATA_DIR / "architecture"]
from xsql.execute import result_match  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    items = {it["idx"]: it for it in read_jsonl(SLICE_DIR / "items.jsonl")}
    cands = read_jsonl(SLICE_DIR / "candidates.jsonl")

    changed: list[dict] = []
    new_label: dict[str, bool] = {}
    for c in cands:
        it = items[c["item_idx"]]
        correct, status = result_match(it["db_id"], c["sql"], it["gold_sql"])
        new_label[c["candidate_id"]] = correct
        if correct != c["correct"]:
            changed.append(
                dict(
                    candidate_id=c["candidate_id"],
                    db_id=c["db_id"],
                    question=it["questions"]["en"],
                    sql=c["sql"],
                    gold_sql=it["gold_sql"],
                    old=c["correct"],
                    new=correct,
                )
            )
        c["correct"] = correct
        c["exec_status"] = status

    write_jsonl(SLICE_DIR / "candidates.jsonl", cands)
    score_files = sorted(SLICE_DIR.glob("scores_*.jsonl"))
    for d in DEPENDENT_DIRS:
        score_files += sorted(d.glob("scores_*.jsonl"))
    for path in score_files:
        rows = read_jsonl(path)
        if not all(r["candidate_id"] in new_label for r in rows):
            print(f"skipping {path}: candidates not from this run")
            continue
        for r in rows:
            r["correct"] = new_label[r["candidate_id"]]
        write_jsonl(path, rows)
        print(f"relabeled {path}")

    n_correct = sum(new_label.values())
    lines = [
        "# Relabeling under a column-order-insensitive comparator",
        "",
        "The original execution-match labels compared result columns positionally.",
        "Spider's evaluator maps columns onto the SELECT list and does not enforce",
        "their order, so every candidate was relabeled with `xsql.execute.rows_match`,",
        "which accepts a prediction if any permutation of its columns matches the gold",
        "result (rows as a multiset, or in order when the gold query has ORDER BY).",
        "",
        f"- Candidates: {len(cands)}",
        f"- Correct before: {sum(1 for c in changed if c['old']) + n_correct - sum(1 for c in changed if c['new'])}",
        f"- Correct after: {n_correct}",
        f"- Labels changed: {len(changed)} "
        f"({sum(1 for c in changed if c['new'])} incorrect->correct, "
        f"{sum(1 for c in changed if not c['new'])} correct->incorrect)",
        f"- Questions affected: {len({c['candidate_id'].split('#')[0] for c in changed})}",
        "",
        "| candidate | db | old | new | candidate SQL | gold SQL |",
        "|---|---|---|---|---|---|",
    ]
    for c in changed:
        sql = c["sql"].replace("|", "\\|")
        gold = c["gold_sql"].replace("|", "\\|")
        lines.append(
            f"| {c['candidate_id']} | {c['db_id']} | {int(c['old'])} | {int(c['new'])} | `{sql}` | `{gold}` |"
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "relabel_column_order.md").write_text("\n".join(lines) + "\n")
    print(f"{len(changed)} labels changed; {n_correct}/{len(cands)} correct")


if __name__ == "__main__":
    main()
