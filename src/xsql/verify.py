"""LLM correctness verifier.

The verifier is the object of study: it sees a fixed (schema, SQL) pair plus a
question rendered in one language, and returns P(the SQL correctly answers the
question). Responses are cached on disk so re-running analysis costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic

from .config import DATA_DIR, VERIFIER_MODEL, require_api_key

CACHE_PATH = DATA_DIR / "verifier_cache.sqlite"

SYSTEM = (
    "You are a meticulous database engineer auditing text-to-SQL output. "
    "You judge whether a candidate SQLite query correctly answers a user's "
    "question against a given schema. The question may be written in any "
    "language; judge the query on its merits, not on the question's language."
)

TEMPLATE = """Database schema:
{schema}

User question:
{question}

Candidate SQLite query:
{sql}

Will this query correctly answer the user's question? Think briefly, then end \
your reply with exactly one line:

CONFIDENCE: <probability between 0.00 and 1.00 that the query is correct>"""

CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)


@dataclass
class Verdict:
    """`confidence` is None when the model never emitted a parseable score.

    Unparsed results are surfaced rather than defaulted to 0.5: a silent 0.5 is
    indistinguishable from genuine uncertainty and quietly corrupts calibration.
    """

    confidence: float | None
    raw: str
    stop_reason: str | None


class VerifierCache:
    """Disk cache keyed by the exact prompt sent to the model."""

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Verification is IO-bound and runs across a thread pool; the lock keeps
        # the shared connection safe.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
            )
            self.conn.commit()

    def get(self, key: str) -> dict | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value))
            )
            self.conn.commit()


def parse_confidence(text: str) -> float | None:
    matches = CONFIDENCE_RE.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    return min(max(value, 0.0), 1.0)


MAX_TOKENS = 1024
# Non-English reasoning runs longer and can be cut off before the score line.
RETRY_MAX_TOKENS = 3072


class Verifier:
    def __init__(self, model: str = VERIFIER_MODEL, cache: VerifierCache | None = None):
        self.model = model
        self.client = Anthropic(api_key=require_api_key())
        self.cache = cache if cache is not None else VerifierCache()

    def _call(self, prompt: str, max_tokens: int) -> dict:
        key = hashlib.sha256(
            f"{self.model}\x00{max_tokens}\x00{prompt}".encode()
        ).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        record = {
            "raw": "".join(b.text for b in response.content if b.type == "text"),
            "stop_reason": response.stop_reason,
        }
        self.cache.put(key, record)
        return record

    def score(self, schema: str, question: str, sql: str) -> Verdict:
        prompt = TEMPLATE.format(schema=schema, question=question, sql=sql)
        record = self._call(prompt, MAX_TOKENS)
        confidence = parse_confidence(record["raw"])

        # A truncated reply loses the trailing score line; retry with more room
        # and an explicit brevity instruction before giving up.
        if confidence is None:
            retry_prompt = prompt + (
                "\n\nKeep your reasoning to at most three sentences. The final "
                "CONFIDENCE line is mandatory."
            )
            record = self._call(retry_prompt, RETRY_MAX_TOKENS)
            confidence = parse_confidence(record["raw"])

        return Verdict(
            confidence=confidence,
            raw=record["raw"],
            stop_reason=record.get("stop_reason"),
        )
