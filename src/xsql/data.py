"""Loading parallel MultiSpider questions and Spider schemas."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import LANGUAGES, db_path, questions_dir


@dataclass
class Item:
    """One underlying question, with its translations and gold SQL.

    MultiSpider's per-language dev files are index-aligned: position i holds the
    same question, db_id, and gold query in every language, so `questions` is
    keyed by language code.
    """

    idx: int
    db_id: str
    gold_sql: str
    questions: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(**d)


def load_parallel(languages: list[str] = LANGUAGES) -> list[Item]:
    """Load the dev split in every language and zip it into parallel items."""
    per_lang = {
        lang: json.loads((questions_dir() / f"dev_{lang}.json").read_text())
        for lang in languages
    }

    lengths = {lang: len(rows) for lang, rows in per_lang.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"dev files are not parallel: {lengths}")

    pivot_rows = per_lang[languages[0]]
    items: list[Item] = []
    for i, base in enumerate(pivot_rows):
        # Guard the alignment assumption rather than trusting it silently.
        if any(per_lang[lang][i]["db_id"] != base["db_id"] for lang in languages):
            raise ValueError(f"db_id mismatch across languages at index {i}")
        items.append(
            Item(
                idx=i,
                db_id=base["db_id"],
                gold_sql=base["query"],
                questions={lang: per_lang[lang][i]["question"] for lang in languages},
            )
        )
    return items


def schema_ddl(db_id: str) -> str:
    """CREATE TABLE statements for a database, used as the prompt schema."""
    with sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    return "\n\n".join(sql.strip() for (sql,) in rows)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
