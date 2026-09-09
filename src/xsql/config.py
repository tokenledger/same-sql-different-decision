"""Paths and slice-level constants."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "dreamerdeo/multispider"

# MultiSpider ships two variants. `with_original_value` localizes the literal
# values inside questions ("JetBlue Airways" -> "深圳航空公司") but leaves the gold
# SQL in English, so translated questions no longer ask what the gold query
# answers. `with_english_value` holds literals fixed, which is what the paired
# design requires: only the language may vary.
VARIANT = "with_english_value"

# Question languages. `en` is the pivot: candidates are generated from the
# English question, then re-scored under every language here.
LANGUAGES = ["en", "de", "zh"]
PIVOT = "en"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
# Which run's data to read/write. Set XSQL_RUN to keep separate runs (the 20-item
# pilot vs a scaled Colab run) from overwriting each other.
SLICE_DIR = DATA_DIR / os.environ.get("XSQL_RUN", "slice")
RESULTS_DIR = PROJECT_ROOT / "results"

GENERATOR_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
VERIFIER_MODEL = "claude-opus-5"
# Deliberately not the generator model: a model scoring its own output shows
# self-preference that would be confounded with any language effect.
LOCAL_VERIFIER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Vertical-slice sizes. Scale these up once the pipeline is proven.
N_QUESTIONS = 20
K_CANDIDATES = 3


@lru_cache(maxsize=1)
def data_root() -> Path:
    """Local snapshot of the MultiSpider repo, downloading it if absent."""
    return Path(
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            tqdm_class=None,
            allow_patterns=[
                f"dataset/multispider/{VARIANT}/dev_*.json",
                "dataset/spider/database/**",
            ],
        )
    )


def questions_dir() -> Path:
    return data_root() / "dataset/multispider" / VARIANT


def database_dir() -> Path:
    return data_root() / "dataset/spider/database"


def db_path(db_id: str) -> Path:
    return database_dir() / db_id / f"{db_id}.sqlite"


def require_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return key
