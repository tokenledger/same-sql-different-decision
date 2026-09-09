"""Robustness of the decision-level results to the operating point and the split.

Two diagnostics requested at review, both computed from saved scores only:

1. Threshold sensitivity (canonical split). The English threshold is refit at
   5%, 10%, and 15% risk targets and the per-language flip rate, unsafe
   promotion, and English coverage are reported at each. Flip rates at a
   lower target are not "more robust" if the verifier is barely accepting
   anything, so English coverage is printed alongside.

2. Score-precision stress test. Stored probabilities and the calibrated
   threshold are rounded to float32 and the decisions recomputed. Qwen's
   threshold sits within a few 1e-6 of the score ceiling, so this shows how
   much of its operating point rests on digits a lower-precision store would
   lose. It is a conversion test, not a new inference run.

3. Repeated database splits. Seeds 0-9 each shuffle the 163 databases,
   fit the English threshold on the first half, and evaluate on the second.
   Reported: mean flip rate across the six non-English languages, per-language
   unsafe promotion (share of all held-out incorrect candidates), and the
   pivot-versus-quantile flip advantage. Ranges across seeds are descriptive;
   splits overlap heavily and are not independent replications.

Run: uv run python scripts/21_robustness.py  ->  results/robustness.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsql.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from xsql.metrics import threshold_at_risk  # noqa: E402

BIG = DATA_DIR / "big"
MIT = DATA_DIR / "mitigation"
VERIFIERS = [("llama8b", "Llama-3.1-8B"), ("qwen7b", "Qwen2.5-7B")]
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGETS = [0.05, 0.10, 0.15]
SEEDS = list(range(10))


def load_scores(path: Path, ids: list[str] | None = None):
    rows = [json.loads(l) for l in path.open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["lang"]][r["candidate_id"]] = r
    return by


def load(verifier: str):
    by = load_scores(BIG / f"scores_big-{verifier}.jsonl")
    ids = sorted(by[PIVOT])
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    conf = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in [PIVOT] + LANGS}
    pivot = load_scores(MIT / f"scores_mitigation_pivot-{verifier}.jsonl")
    conf_pivot = {l: np.array([pivot[l][c]["confidence"] for c in ids]) for l in LANGS}
    return labels, dbs, conf, conf_pivot


def split_masks(dbs: np.ndarray, seed: int):
    uniq = sorted(set(dbs))
    perm = np.random.default_rng(seed).permutation(uniq)
    fit = set(perm[: len(uniq) // 2])
    m_fit = np.array([d in fit for d in dbs])
    return m_fit, ~m_fit


def quantile_map(src_cal: np.ndarray, en_cal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map language scores onto English's calibration-split quantiles (as in 13_transport)."""
    q = np.searchsorted(np.sort(src_cal), x, side="left") / len(src_cal)
    return np.quantile(en_cal, np.clip(q, 0, 1))


def decisions(labels, conf, m_fit, m_test, target):
    thr = threshold_at_risk(labels[m_fit], conf[PIVOT][m_fit], target)
    lab = labels[m_test]
    acc = {l: conf[l][m_test] >= thr for l in conf}
    return thr, lab, acc


def main() -> None:
    lines = [
        "# Robustness to the operating point and to the database split",
        "",
        "Same saved scores and labels as the main analysis; no new inference.",
        "",
        "## 1. Threshold sensitivity, canonical split (seed 0)",
        "",
        "English threshold refit at each risk target on the 81 calibration databases,",
        "then applied unchanged to every language on the 82 test databases. A target",
        "at or above the verifier's unconditional English error rate on the",
        "calibration split is met by accepting nearly everything, so flips vanish",
        "trivially there; English coverage is the column to read first.",
        "",
    ]
    for verifier, pretty in VERIFIERS:
        labels, dbs, conf, _ = load(verifier)
        m_fit, m_test = split_masks(dbs, 0)
        lines += [
            f"### {pretty}",
            "",
            "| target | threshold | en coverage | en risk | flip range (6 langs) | mean flip | unsafe promotion range (of all incorrect) |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in TARGETS:
            thr, lab, acc = decisions(labels, conf, m_fit, m_test, t)
            flips = [float((acc[PIVOT] != acc[l]).mean()) for l in LANGS]
            n_inc = (lab == 0).sum()
            unsafe = [float((~acc[PIVOT] & acc[l] & (lab == 0)).sum() / n_inc) for l in LANGS]
            en_cov = float(acc[PIVOT].mean())
            en_risk = float((lab[acc[PIVOT]] == 0).mean()) if acc[PIVOT].any() else float("nan")
            lines.append(
                f"| {t:.0%} | {thr:.6g} | {en_cov:.1%} | {en_risk:.3f} | "
                f"{min(flips):.1%}-{max(flips):.1%} | {np.mean(flips):.1%} | "
                f"{min(unsafe):.1%}-{max(unsafe):.1%} |"
            )
        lines.append("")

    lines += [
        "## 2. Score-precision stress test (canonical split, 10% target)",
        "",
        "Decisions recomputed after rounding every stored probability and the fixed",
        "calibrated threshold to float32. `changed` is the share of (candidate,",
        "language) decisions on the test split that differ from the full-precision",
        "policy. This is a storage-precision conversion test, not a new fp32 model run.",
        "",
        "| verifier | threshold | float32 threshold | en coverage | en coverage (f32) | decisions changed (7 langs) | distinct en scores | distinct en scores (f32) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for verifier, pretty in VERIFIERS:
        labels, dbs, conf, _ = load(verifier)
        m_fit, m_test = split_masks(dbs, 0)
        thr, lab, acc = decisions(labels, conf, m_fit, m_test, 0.10)
        thr32 = float(np.float32(thr))
        acc32 = {l: conf[l][m_test].astype(np.float32) >= np.float32(thr) for l in conf}
        changed = np.mean([float((acc[l] != acc32[l]).mean()) for l in conf])
        lines.append(
            f"| {pretty} | {thr!r} | {thr32!r} | {acc[PIVOT].mean():.1%} | {acc32[PIVOT].mean():.1%} | "
            f"{changed:.1%} | {len(np.unique(conf[PIVOT]))} | {len(np.unique(conf[PIVOT].astype(np.float32)))} |"
        )
    lines.append("")

    lines += [
        "## 3. Repeated database splits (seeds 0-9)",
        "",
        "Each seed reshuffles the 163 databases; the English threshold is refit at the",
        "10% target on the first 81 and evaluated on the remaining 82. Seed 0 is the",
        "canonical split used in the paper. Ranges are descriptive, not confidence",
        "intervals: the splits overlap and are not independent replications.",
        "",
    ]
    for verifier, pretty in VERIFIERS:
        labels, dbs, conf, conf_pivot = load(verifier)
        per_seed = []
        for s in SEEDS:
            m_fit, m_test = split_masks(dbs, s)
            thr, lab, acc = decisions(labels, conf, m_fit, m_test, 0.10)
            n_inc = (lab == 0).sum()
            flips = {l: float((acc[PIVOT] != acc[l]).mean()) for l in LANGS}
            unsafe = {l: float((~acc[PIVOT] & acc[l] & (lab == 0)).sum() / n_inc) for l in LANGS}
            # quantile transport and pivot, both compared with the English decision
            q_flips, p_flips = [], []
            for l in LANGS:
                mapped = quantile_map(conf[l][m_fit], conf[PIVOT][m_fit], conf[l][m_test])
                q_flips.append(float(((mapped >= thr) != acc[PIVOT]).mean()))
                p_flips.append(float(((conf_pivot[l][m_test] >= thr) != acc[PIVOT]).mean()))
            per_seed.append(
                dict(seed=s, thr=thr, flips=flips, unsafe=unsafe,
                     en_cov=float(acc[PIVOT].mean()),
                     mean_flip=float(np.mean(list(flips.values()))),
                     quantile=float(np.mean(q_flips)), pivot=float(np.mean(p_flips)))
            )
        lines += [
            f"### {pretty}",
            "",
            "| seed | threshold | en coverage | mean flip (6 langs) | worst-lang flip | quantile flip | pivot flip | pivot - quantile | "
            + " | ".join(f"unsafe {l}" for l in LANGS) + " |",
            "|---|---|---|---|---|---|---|---|" + "---|" * len(LANGS),
        ]
        for r in per_seed:
            worst = max(r["flips"], key=r["flips"].get)
            lines.append(
                f"| {r['seed']} | {r['thr']:.6g} | {r['en_cov']:.1%} | {r['mean_flip']:.1%} | "
                f"{r['flips'][worst]:.1%} ({worst}) | {r['quantile']:.1%} | {r['pivot']:.1%} | "
                f"{100 * (r['pivot'] - r['quantile']):+.1f} pp | "
                + " | ".join(f"{r['unsafe'][l]:.1%}" for l in LANGS) + " |"
            )
        mf = [r["mean_flip"] for r in per_seed]
        adv = [100 * (r["pivot"] - r["quantile"]) for r in per_seed]
        lines += [
            "",
            f"Mean flip across seeds: {min(mf):.1%}-{max(mf):.1%} (mean {np.mean(mf):.1%}, sd {np.std(mf):.1%}). "
            f"Pivot minus quantile: {min(adv):+.1f} to {max(adv):+.1f} pp "
            f"(pivot flips fewer decisions in {sum(a < 0 for a in adv)}/{len(adv)} splits).",
            "",
            "Unsafe promotion range across seeds (share of all held-out incorrect candidates):",
            "",
            "| lang | min | max |",
            "|---|---|---|",
        ]
        for l in LANGS:
            u = [r["unsafe"][l] for r in per_seed]
            lines.append(f"| {l} | {min(u):.1%} | {max(u):.1%} |")
        lines.append("")

    out = RESULTS_DIR / "robustness.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
