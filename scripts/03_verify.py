"""Paired verifier experiment.

For every candidate SQL (all generated from the English question), score the
*same* SQL under each language's rendering of the question. Schema, SQL,
database, and correctness label are held fixed; only the question language
changes.

Writes data/slice/scores.jsonl.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

from xsql.config import LANGUAGES, SLICE_DIR
from xsql.data import Item, read_jsonl, schema_ddl, write_jsonl

MAX_WORKERS = 8


def main(backend: str = "api") -> None:
    items = {d["idx"]: Item.from_dict(d) for d in read_jsonl(SLICE_DIR / "items.jsonl")}
    candidates = read_jsonl(SLICE_DIR / "candidates.jsonl")
    schemas = {db_id: schema_ddl(db_id) for db_id in {c["db_id"] for c in candidates}}

    jobs = [(c, lang) for c in candidates for lang in LANGUAGES]
    print(f"{len(candidates)} candidates x {len(LANGUAGES)} languages = {len(jobs)} calls")

    # Backend label also names the output file, so runs from different models
    # sit side by side instead of overwriting each other.
    if backend.startswith("local"):
        from xsql.verify_local import LocalVerifier

        verifier = LocalVerifier()
        # A local model holds one MPS context; threading it would not help.
        workers = 1
        print(f"backend: local ({verifier.model_name})")
    else:
        from xsql.verify import Verifier

        verifier = Verifier()
        workers = MAX_WORKERS
        print(f"backend: api ({verifier.model})")

    def run(job) -> dict:
        cand, lang = job
        item = items[cand["item_idx"]]
        verdict = verifier.score(
            schema=schemas[cand["db_id"]],
            question=item.questions[lang],
            sql=cand["sql"],
        )
        return {
            "candidate_id": cand["candidate_id"],
            "item_idx": cand["item_idx"],
            "db_id": cand["db_id"],
            "lang": lang,
            "correct": cand["correct"],
            "confidence": verdict.confidence,
            "stop_reason": verdict.stop_reason,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = []
        for n, rec in enumerate(pool.map(run, jobs), 1):
            scores.append(rec)
            if n % 20 == 0 or n == len(jobs):
                print(f"  scored {n}/{len(jobs)}", flush=True)

    write_jsonl(SLICE_DIR / f"scores_{backend}.jsonl", scores)
    unparsed = [s for s in scores if s["confidence"] is None]
    print(f"wrote {len(scores)} scores ({len(unparsed)} without a parseable confidence)")
    if unparsed:
        for lang in LANGUAGES:
            n = sum(1 for s in unparsed if s["lang"] == lang)
            if n:
                print(f"  {lang}: {n} unparsed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "api")
