"""Significance analysis with cluster-bootstrapped confidence intervals.

Resampling is over the underlying questions, not over candidates: the K
candidates from one question share a database, a schema, and a difficulty, so
treating them as independent would understate every interval.

Usage:  uv run python scripts/05_analyze.py [backend ...]
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from xsql.config import LANGUAGES, PIVOT, RESULTS_DIR, SLICE_DIR
from xsql.data import read_jsonl
from xsql.metrics import auroc, risk_at_threshold, threshold_at_risk

N_BOOT = 4000
TARGET_RISK = 0.10
SEED = 0


def load(backend: str):
    scores = read_jsonl(SLICE_DIR / f"scores_{backend}.jsonl")
    by_lang: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in scores:
        by_lang[r["lang"]][r["candidate_id"]] = r

    langs = [l for l in LANGUAGES if l in by_lang]
    langs += sorted(set(by_lang) - set(langs))

    # Candidates usable in every language, so all languages share one sample.
    ids = sorted(
        c
        for c in by_lang[PIVOT]
        if all(by_lang[l].get(c, {}).get("confidence") is not None for l in langs)
    )
    # Cluster on the underlying question. Older runs used numeric ids embedded in
    # the candidate id; newer ones carry item_idx (e.g. "train:812") explicitly.
    item_key = {
        c: by_lang[PIVOT][c].get("item_idx", c.rsplit(":", 1)[0]) for c in ids
    }
    codes = {k: n for n, k in enumerate(sorted(set(item_key.values()), key=str))}
    items = np.array([codes[item_key[c]] for c in ids])
    labels = np.array([by_lang[PIVOT][c]["correct"] for c in ids], dtype=float)
    conf = {l: np.array([by_lang[l][c]["confidence"] for c in ids]) for l in langs}
    return langs, items, labels, conf


def bootstrap(items: np.ndarray, stat, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI, resampling whole questions with replacement."""
    uniq = np.unique(items)
    index_of = {i: np.flatnonzero(items == i) for i in uniq}
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([index_of[i] for i in pick])
        v = stat(idx)
        if v is not None and np.isfinite(v):
            out.append(v)
    if not out:
        return float("nan"), float("nan")
    return tuple(np.percentile(out, [2.5, 97.5]))


def analyze(backend: str) -> list[str]:
    langs, items, labels, conf = load(backend)
    rng = np.random.default_rng(SEED)
    n_items = len(np.unique(items))

    lines = [
        f"## {backend}",
        "",
        f"{len(labels)} candidates from {n_items} questions, "
        f"{int(labels.sum())} correct ({labels.mean():.1%}), "
        f"{int((1 - labels).sum())} incorrect. "
        f"CIs are percentile bootstrap over questions ({N_BOOT} resamples).",
        "",
        "### Paired confidence shift vs. English (identical SQL and label)",
        "",
        "| lang | mean Δconf | 95% CI | significant |",
        "|---|---|---|---|",
    ]
    for lang in langs:
        if lang == PIVOT:
            continue
        d = conf[lang] - conf[PIVOT]
        lo, hi = bootstrap(items, lambda i, d=d: d[i].mean(), rng)
        sig = "**yes**" if lo > 0 or hi < 0 else "no"
        lines.append(f"| {lang} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {sig} |")

    lines += [
        "",
        "### AUROC gap vs. English",
        "",
        "| lang | AUROC | gap en−lang | 95% CI | significant |",
        "|---|---|---|---|---|",
    ]

    def gap(i, lang):
        a, b = auroc(labels[i], conf[PIVOT][i]), auroc(labels[i], conf[lang][i])
        return None if a is None or b is None else a - b

    for lang in langs:
        a = auroc(labels, conf[lang])
        if lang == PIVOT:
            lines.append(f"| {lang} | {a:.3f} | — | — | — |")
            continue
        lo, hi = bootstrap(items, lambda i, l=lang: gap(i, l), rng)
        sig = "**yes**" if lo > 0 or hi < 0 else "no"
        d = auroc(labels, conf[PIVOT]) - a
        lines.append(f"| {lang} | {a:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |")

    thr = threshold_at_risk(labels, conf[PIVOT], TARGET_RISK)
    lines += [
        "",
        f"### Coverage under the English {TARGET_RISK:.0%}-risk threshold "
        f"({'n/a' if thr is None else f'{thr:.4f}'})",
        "",
        "| lang | realized risk | coverage | gap en−lang | 95% CI | significant |",
        "|---|---|---|---|---|---|",
    ]
    if thr is None:
        lines.append("| — | no threshold reaches the target risk | | | | |")
        return lines

    def cov_gap(i, lang):
        return (
            risk_at_threshold(labels[i], conf[PIVOT][i], thr)[1]
            - risk_at_threshold(labels[i], conf[lang][i], thr)[1]
        )

    for lang in langs:
        risk, cov = risk_at_threshold(labels, conf[lang], thr)
        if lang == PIVOT:
            lines.append(f"| {lang} | {risk:.3f} | {cov:.3f} | — | — | — |")
            continue
        lo, hi = bootstrap(items, lambda i, l=lang: cov_gap(i, l), rng)
        sig = "**yes**" if lo > 0 or hi < 0 else "no"
        d = risk_at_threshold(labels, conf[PIVOT], thr)[1] - cov
        lines.append(
            f"| {lang} | {risk:.3f} | {cov:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |"
        )
    return lines


def main(backends: list[str]) -> None:
    out = ["# Cross-lingual verifier significance analysis", ""]
    for b in backends:
        out += analyze(b) + [""]
    text = "\n".join(out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "significance.md").write_text(text)
    print(text)


if __name__ == "__main__":
    args = sys.argv[1:] or ["api", "local-qwen1.5b", "local-llama8b"]
    main(args)
