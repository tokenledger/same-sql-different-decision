"""Build the vertical-slice item set.

Samples N_QUESTIONS parallel MultiSpider dev questions whose gold SQL executes
cleanly, spread across distinct databases so the slice is not dominated by one
schema. Writes data/slice/items.jsonl.
"""

from __future__ import annotations

import random
from collections import defaultdict

from xsql.config import N_QUESTIONS, SLICE_DIR
from xsql.data import load_parallel, write_jsonl
from xsql.execute import execute

SEED = 0
MAX_PER_DB = 2


def main() -> None:
    items = load_parallel()
    print(f"loaded {len(items)} parallel dev items")

    rng = random.Random(SEED)
    order = list(items)
    rng.shuffle(order)

    per_db: dict[str, int] = defaultdict(int)
    chosen = []
    skipped = 0
    for item in order:
        if len(chosen) >= N_QUESTIONS:
            break
        if per_db[item.db_id] >= MAX_PER_DB:
            continue
        gold = execute(item.db_id, item.gold_sql)
        if not gold.ok:
            skipped += 1
            continue
        # Degenerate empty gold results make execution match trivially satisfiable.
        if not gold.rows:
            skipped += 1
            continue
        per_db[item.db_id] += 1
        chosen.append(item)

    write_jsonl(SLICE_DIR / "items.jsonl", [i.to_dict() for i in chosen])
    print(f"selected {len(chosen)} items across {len(per_db)} databases "
          f"({skipped} skipped for unusable gold)")
    for item in chosen[:3]:
        print(f"  [{item.db_id}] {item.questions['en']}")


if __name__ == "__main__":
    main()
