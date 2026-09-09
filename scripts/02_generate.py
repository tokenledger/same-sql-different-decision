"""Generate K SQL candidates per slice item from the English question, then
execute each one to attach a correctness label.

Candidates are generated from English only: the paired experiment holds the SQL
fixed and varies only the language the verifier sees.

Writes data/slice/candidates.jsonl.
"""

from __future__ import annotations

import time
from collections import Counter

from xsql.config import K_CANDIDATES, PIVOT, SLICE_DIR
from xsql.data import Item, read_jsonl, schema_ddl, write_jsonl
from xsql.execute import result_match
from xsql.generate import generate_candidates


def main() -> None:
    items = [Item.from_dict(d) for d in read_jsonl(SLICE_DIR / "items.jsonl")]
    print(f"{len(items)} items x K={K_CANDIDATES} candidates")

    records = []
    status_counts: Counter[str] = Counter()
    start = time.monotonic()

    for n, item in enumerate(items, 1):
        schema = schema_ddl(item.db_id)
        sqls = generate_candidates(
            schema, item.questions[PIVOT], k=K_CANDIDATES, seed=item.idx
        )
        for j, sql in enumerate(sqls):
            correct, status = result_match(item.db_id, sql, item.gold_sql)
            status_counts[status] += 1
            records.append(
                {
                    "candidate_id": f"{item.idx}:{j}",
                    "item_idx": item.idx,
                    "db_id": item.db_id,
                    "sql": sql,
                    "correct": correct,
                    "exec_status": status,
                }
            )
        n_correct = sum(r["correct"] for r in records)
        elapsed = time.monotonic() - start
        print(
            f"[{n}/{len(items)}] {item.db_id}: "
            f"{sum(r['correct'] for r in records[-K_CANDIDATES:])}/{K_CANDIDATES} correct "
            f"| running {n_correct}/{len(records)} | {elapsed / n:.1f}s per item",
            flush=True,
        )

    write_jsonl(SLICE_DIR / "candidates.jsonl", records)
    n_correct = sum(r["correct"] for r in records)
    print(
        f"\n{len(records)} candidates: {n_correct} correct, "
        f"{len(records) - n_correct} incorrect"
    )
    print("execution status:", dict(status_counts))


if __name__ == "__main__":
    main()
