"""Analysis for the ARCHITECTURE experiment: label-word robustness, dense vs
sparse-MoE, and the multilingual training-distribution control.

Consumes:
  - data/big/scores_big-{llama8b,qwen7b}.jsonl        (existing, Yes/No, full 1200 items)
  - data/architecture/scores_architecture_{label_scheme}-{verifier}.jsonl
    (new, written by the notebook emitted by scripts/build_architecture_notebook.py,
    300-item database-disjoint subset; label_scheme in {yesno, correctincorrect};
    verifier in {llama8b, qwen7b, qwen3-4b-dense, qwen3-30ba3b-moe, aya-expanse-8b})

Every discovered (verifier, label_scheme) dataset is analyzed with the same
methodology as scripts/12_decision_analysis.py and scripts/13_transport.py, so
numbers are directly comparable: fit an English threshold at TARGET_RISK on a
database-disjoint half of the item pool, apply it unchanged to every language
on the other half, and report what that does to individual execute/defer
decisions on identical SQL.

Run against only the two `big` score files, the script reproduces the headline
numbers (Llama-3.1-8B mean flips ~14.2%, Qwen2.5-7B mean flips ~19.7%, both
Yes/No, full 1200 items) using the seed=0 database split of
scripts/12_decision_analysis.py. Any `data/architecture/*.jsonl` files found
are analyzed the same way and appended to the same report.

Per (verifier, label_scheme) this reports: English AUROC, saturation fraction
(share of English scores within 1e-6 of {0,1}), mean decision-flip rate vs
English (with a database-clustered bootstrap 95% CI), unsafe-promotion rate
(incorrect SQL promoted into execution, with a Clopper-Pearson 95% CI, since a
percentile bootstrap collapses to [0,0] whenever zero events are observed),
lost-automation rate (correct SQL demoted), accepted-set Jaccard overlap vs
English, coverage spread across languages, and accepted risk per language.
Effects that are statistically significant but smaller than NEGLIGIBLE_ABS are
flagged as negligible.

Usage: uv run python scripts/14_architecture.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import auroc as auroc_metric
from xsql.metrics import threshold_at_risk

BIG = DATA_DIR / "big"
ARCH = DATA_DIR / "architecture"

LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
N_BOOT = 2000
SEED = 0
SAT_EPS = 1e-6
NEGLIGIBLE_ABS = 0.01  # a CI-significant effect smaller than this (1pp) is flagged as negligible

# Existing verifiers, established Yes/No scoring, full 1200-item pool.
EXISTING_DATASETS = [
    ("llama8b", "yesno", BIG / "scores_big-llama8b.jsonl", "Llama-3.1-8B (established, full 1200)"),
    ("qwen7b", "yesno", BIG / "scores_big-qwen7b.jsonl", "Qwen2.5-7B (established, full 1200)"),
]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval; unlike a bootstrap it stays informative at k=0."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def discover_architecture_datasets() -> list[tuple[str, str, Path, str]]:
    """Find every data/architecture/scores_architecture_{scheme}-{verifier}.jsonl."""
    out = []
    if not ARCH.exists():
        return out
    for p in sorted(ARCH.glob("scores_architecture_*.jsonl")):
        # filename: scores_architecture_{label_scheme}-{verifier}.jsonl
        stem = p.stem[len("scores_architecture_"):]
        if "-" not in stem:
            continue
        label_scheme, verifier = stem.split("-", 1)
        out.append((verifier, label_scheme, p, f"{verifier} ({label_scheme}, 300-item subset)"))
    return out


def load(path: Path):
    rows = [json.loads(l) for l in path.open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["lang"]][r["candidate_id"]] = r
    if PIVOT not in by:
        raise ValueError(f"{path}: no '{PIVOT}' rows found")
    ids = sorted(by[PIVOT])
    present_langs = [l for l in [PIVOT] + LANGS if l in by]
    missing = [c for c in ids if any(c not in by[l] for l in present_langs)]
    if missing:
        ids = [c for c in ids if c not in set(missing)]
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    conf = {l: np.array([by[l][c]["confidence"] for c in ids], dtype=float) for l in present_langs}
    return labels, dbs, conf, present_langs


def analyze(name: str, label_scheme: str, path: Path, pretty: str, lines: list[str]) -> None:
    labels, dbs, conf, present_langs = load(path)
    langs_here = [l for l in LANGS if l in present_langs]

    # English AUROC and saturation are computed on all items, not only the test split.
    en_conf = conf[PIVOT]
    en_valid = ~np.isnan(en_conf)
    en_auroc = auroc_metric(labels[en_valid], en_conf[en_valid])
    sat_frac = float(
        ((np.abs(en_conf[en_valid]) < SAT_EPS) | (np.abs(1 - en_conf[en_valid]) < SAT_EPS)).mean()
    )

    lines += [
        f"## {pretty}",
        "",
        f"n={len(labels)} candidates across {len(set(dbs))} databases, "
        f"{int(labels.sum())} correct / {int((1 - labels).sum())} incorrect "
        f"({labels.mean():.1%} correct). Languages present: {', '.join(present_langs)}.",
        "",
        f"**English AUROC:** {en_auroc:.4f}" if en_auroc is not None else "**English AUROC:** undefined (single class)",
        f"  **Saturation fraction (within 1e-6 of 0 or 1):** {sat_frac:.1%} "
        f"{'(majority saturated; treat isotonic-style transport with caution)' if sat_frac >= 0.5 else '(not saturated)'}",
        "",
    ]

    if not langs_here:
        lines.append("_No non-English languages present in this file; decision-flip analysis skipped._\n")
        return
    if en_auroc is None or len(set(dbs)) < 4:
        lines.append("_Too few databases or a degenerate label distribution for a fit/test split; skipped._\n")
        return

    # Database-disjoint fit/test split with the same seed as
    # scripts/12_decision_analysis.py, so the `big` datasets reproduce its numbers.
    uniq = sorted(set(dbs))
    perm = np.random.default_rng(SEED).permutation(uniq)
    fit = set(perm[: len(uniq) // 2])
    m_fit = np.array([d in fit for d in dbs])
    m_test = ~m_fit

    fit_valid = m_fit & en_valid
    thr = threshold_at_risk(labels[fit_valid], en_conf[fit_valid], TARGET_RISK)
    if thr is None:
        lines.append("_No feasible English threshold at the target risk on the fit split; skipped._\n")
        return

    lab = labels[m_test]
    dbs_t = dbs[m_test]
    uniq_t = np.unique(dbs_t)
    idx_of = {d: np.flatnonzero(dbs_t == d) for d in uniq_t}
    rng = np.random.default_rng(SEED)

    def cluster_bootstrap_mean(values_by_lang: dict[str, np.ndarray]) -> tuple[float, float]:
        """95% percentile CI on the mean-over-languages flip rate, resampling
        databases (not candidates) with replacement so within-database
        correlation does not understate the interval."""
        means = []
        for _ in range(N_BOOT):
            sampled_dbs = rng.choice(uniq_t, len(uniq_t), replace=True)
            idx = np.concatenate([idx_of[d] for d in sampled_dbs])
            means.append(np.mean([values_by_lang[l][idx].mean() for l in values_by_lang]))
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    acc = {}
    valid_test = {}
    for l in [PIVOT] + langs_here:
        c = conf[l][m_test]
        v = ~np.isnan(c)
        valid_test[l] = v
        a = np.zeros(len(c), dtype=bool)
        a[v] = c[v] >= thr
        acc[l] = a

    n_corr, n_incorr = int(lab.sum()), int((1 - lab).sum())
    lines += [
        f"Threshold {thr:.4f} (fit on English, {len(fit)}/{len(uniq)} databases, target risk {TARGET_RISK:.0%}). "
        f"Held-out test split: {len(lab)} candidates, {n_corr} correct / {n_incorr} incorrect.",
        "",
        "| lang | flip rate | unsafe promoted (95% CI) | lost automation | Jaccard overlap | coverage | accepted risk |",
        "|---|---|---|---|---|---|---|",
    ]

    flip_by_lang = {}
    for l in langs_here:
        flip = acc[PIVOT] != acc[l]
        flip_by_lang[l] = flip.astype(float)
        demoted_corr = int((acc[PIVOT] & ~acc[l] & (lab == 1)).sum())
        promoted_incorr = int((~acc[PIVOT] & acc[l] & (lab == 0)).sum())
        lo_cp, hi_cp = clopper_pearson(promoted_incorr, n_incorr)
        ci = (
            f"0/{n_incorr}, 95% upper bound {hi_cp:.1%}"
            if promoted_incorr == 0
            else f"{promoted_incorr}/{n_incorr}={promoted_incorr/n_incorr:.1%} [{lo_cp:.1%}, {hi_cp:.1%}]"
        )
        inter = float((acc[l] & acc[PIVOT]).sum())
        union = float((acc[l] | acc[PIVOT]).sum())
        jac = inter / union if union else float("nan")
        cov = float(acc[l].mean())
        risk = float(1 - lab[acc[l]].mean()) if acc[l].any() else float("nan")
        lines.append(
            f"| {l} | {flip.mean():.1%} | {ci} | "
            f"{demoted_corr}/{n_corr}={demoted_corr/n_corr:.1%} | {jac:.3f} | {cov:.3f} | {risk:.3f} |"
        )

    mean_flip = float(np.mean([flip_by_lang[l].mean() for l in langs_here]))
    lo, hi = cluster_bootstrap_mean(flip_by_lang)
    coverages = [float(acc[l].mean()) for l in langs_here] + [float(acc[PIVOT].mean())]
    cov_spread = max(coverages) - min(coverages)

    negligible_note = ""
    if lo > 0 and mean_flip < NEGLIGIBLE_ABS:
        negligible_note = " (statistically significant but < 1pp, practically negligible)"

    lines += [
        "",
        f"**Mean flip rate across {len(langs_here)} non-English languages: {mean_flip:.1%} "
        f"(database-clustered 95% CI [{lo:.1%}, {hi:.1%}]){negligible_note}.**",
        f"**Coverage spread across languages (incl. English): {cov_spread:.3f}.**",
        "",
    ]


def analyze_pairwise(
    label_a: str, path_a: Path, label_b: str, path_b: Path, comparison_label: str, lines: list[str]
) -> None:
    """Compare two verifier/precision conditions directly against each other
    rather than each against its own English baseline. Used for two pairs:

      - quantization-only: qwen3-4b-bf16 vs qwen3-4b-nf4 (architecture fixed,
        precision varies). Isolates how much 4-bit quantization alone perturbs
        cross-lingual decision consistency.
      - dense vs MoE, precision-matched: qwen3-4b vs qwen3-30ba3b-moe at
        whichever precision the MoE ran at (architecture varies, precision
        fixed).

    A threshold is fit on condition A's English scores (database-disjoint
    fit/test split, same seed as `analyze`), then applied to both A and B in
    every language present in both files. English is included here, unlike in
    the per-verifier tables, because the English column isolates the effect of
    precision or architecture with language held fixed.
    """
    labels_a, dbs_a, conf_a, langs_a = load(path_a)
    labels_b, dbs_b, conf_b, langs_b = load(path_b)

    ids_match = (
        len(labels_a) == len(labels_b)
        and bool((labels_a == labels_b).all())
        and list(dbs_a) == list(dbs_b)
    )
    langs_common = [l for l in [PIVOT] + LANGS if l in langs_a and l in langs_b]

    lines += [f"## Pairwise: {comparison_label}", ""]
    if not ids_match:
        lines.append(
            "_Candidate sets or labels differ between the two files, so a paired "
            "comparison is not possible; skipped._\n"
        )
        return
    if not langs_common:
        lines.append("_No language present in both files; skipped._\n")
        return

    labels, dbs = labels_a, dbs_a
    uniq = sorted(set(dbs))
    if len(uniq) < 4:
        lines.append("_Too few databases for a fit/test split; skipped._\n")
        return
    perm = np.random.default_rng(SEED).permutation(uniq)
    fit = set(perm[: len(uniq) // 2])
    m_fit = np.array([d in fit for d in dbs])
    m_test = ~m_fit

    en_a_fit = conf_a[PIVOT][m_fit]
    fit_valid = ~np.isnan(en_a_fit)
    thr = threshold_at_risk(labels[m_fit][fit_valid], en_a_fit[fit_valid], TARGET_RISK)
    if thr is None:
        lines.append("_No feasible threshold fit on condition A's English scores; skipped._\n")
        return

    lab = labels[m_test]
    n_corr, n_incorr = int(lab.sum()), int((1 - lab).sum())
    dbs_t = dbs[m_test]
    uniq_t = np.unique(dbs_t)
    idx_of = {d: np.flatnonzero(dbs_t == d) for d in uniq_t}
    rng = np.random.default_rng(SEED)

    lines += [
        f"Threshold {thr:.4f} fit on **{label_a}**'s English scores "
        f"({len(fit)}/{len(uniq)} databases, target risk {TARGET_RISK:.0%}), applied "
        f"unchanged to both **{label_a}** and **{label_b}**. Held-out test split: "
        f"{len(lab)} candidates ({n_corr} correct / {n_incorr} incorrect).",
        "",
        "| lang | flip rate (A vs B) | unsafe promoted (95% CI) | lost automation | Jaccard overlap | coverage A | coverage B |",
        "|---|---|---|---|---|---|---|",
    ]

    flip_by_lang = {}
    for l in langs_common:
        ca, cb = conf_a[l][m_test], conf_b[l][m_test]
        va, vb = ~np.isnan(ca), ~np.isnan(cb)
        acc_a = np.zeros(len(ca), dtype=bool); acc_a[va] = ca[va] >= thr
        acc_b = np.zeros(len(cb), dtype=bool); acc_b[vb] = cb[vb] >= thr

        flip = acc_a != acc_b
        flip_by_lang[l] = flip.astype(float)
        demoted = int((acc_a & ~acc_b & (lab == 1)).sum())   # accepted under A, rejected under B
        promoted = int((~acc_a & acc_b & (lab == 0)).sum())  # rejected under A, accepted under B
        lo_cp, hi_cp = clopper_pearson(promoted, n_incorr)
        ci = (
            f"0/{n_incorr}, 95% upper bound {hi_cp:.1%}"
            if promoted == 0
            else f"{promoted}/{n_incorr}={promoted/n_incorr:.1%} [{lo_cp:.1%}, {hi_cp:.1%}]"
        )
        inter = float((acc_a & acc_b).sum())
        union = float((acc_a | acc_b).sum())
        jac = inter / union if union else float("nan")
        lines.append(
            f"| {l} | {flip.mean():.1%} | {ci} | "
            f"{demoted}/{n_corr}={demoted/n_corr:.1%} | {jac:.3f} | {acc_a.mean():.3f} | {acc_b.mean():.3f} |"
        )

    mean_flip = float(np.mean([flip_by_lang[l].mean() for l in langs_common]))

    def cluster_bootstrap_mean(values_by_lang):
        means = []
        for _ in range(N_BOOT):
            sampled_dbs = rng.choice(uniq_t, len(uniq_t), replace=True)
            idx = np.concatenate([idx_of[d] for d in sampled_dbs])
            means.append(np.mean([values_by_lang[l][idx].mean() for l in values_by_lang]))
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    lo, hi = cluster_bootstrap_mean(flip_by_lang)
    negligible_note = ""
    if lo > 0 and mean_flip < NEGLIGIBLE_ABS:
        negligible_note = " (statistically significant but < 1pp, practically negligible)"
    lines += [
        "",
        f"**Mean flip rate across {len(langs_common)} languages (incl. English): {mean_flip:.1%} "
        f"(database-clustered 95% CI [{lo:.1%}, {hi:.1%}]){negligible_note}.**",
        "",
    ]


def find_pairwise_comparisons(new_datasets):
    """Discover the quantization-only and precision-matched dense-vs-MoE pairs
    from whatever architecture score files are present, whichever precision
    the MoE ran at."""
    by_name_scheme = {(n, s): p for n, s, p, _ in new_datasets}
    pairs = []

    dense_bf16 = by_name_scheme.get(("qwen3-4b-bf16", "yesno"))
    dense_nf4 = by_name_scheme.get(("qwen3-4b-nf4", "yesno"))
    if dense_bf16 and dense_nf4:
        pairs.append(("qwen3-4b-bf16", dense_bf16, "qwen3-4b-nf4", dense_nf4,
                       "quantization-only (qwen3-4b, bf16 vs nf4, architecture fixed)"))

    moe_entry = next(
        ((n, s, p) for (n, s), p in by_name_scheme.items() if n.startswith("qwen3-30ba3b-moe-")),
        None,
    )
    if moe_entry:
        moe_name, moe_scheme, moe_path = moe_entry
        moe_precision = moe_name.rsplit("-", 1)[-1]
        dense_match = by_name_scheme.get((f"qwen3-4b-{moe_precision}", moe_scheme))
        if dense_match:
            pairs.append((
                f"qwen3-4b-{moe_precision}", dense_match, moe_name, moe_path,
                f"dense vs sparse-MoE, precision-matched at {moe_precision} "
                f"(architecture varies, precision fixed)",
            ))
    return pairs


def main() -> None:
    lines = [
        "# Architecture and label-word robustness: decision-level results",
        "",
        "One threshold fit on English (database-disjoint half of the item pool),",
        "applied unchanged to every language on the other half. Same SQL, same",
        "database, same execution-derived label; only the question's language",
        "(and, for the label-word conditions, the verdict tokens shown to the",
        "model) differs, so every flip below is attributable to that alone.",
        "",
        "This report is regenerated by `scripts/14_architecture.py` from whatever",
        "score files currently exist: the two established `data/big/scores_big-*.jsonl`",
        "files are always included; any `data/architecture/scores_architecture_*.jsonl`",
        "files (written by the notebook from `scripts/build_architecture_notebook.py`)",
        "are picked up automatically once they exist.",
        "",
    ]

    datasets = list(EXISTING_DATASETS)
    new_datasets = discover_architecture_datasets()
    datasets += new_datasets

    if not new_datasets:
        lines += [
            "**No `data/architecture/*.jsonl` files found.** This run only",
            "reproduces the established Yes/No results from `data/big/` as a",
            "correctness check of this script. Re-run after the architecture notebook",
            "has produced new score files.",
            "",
        ]

    for name, label_scheme, path, pretty in datasets:
        if not path.exists():
            lines.append(f"## {pretty}\n\n_File not found: {path}_\n")
            continue
        analyze(name, label_scheme, path, pretty, lines)

    pairwise = find_pairwise_comparisons(new_datasets)
    if pairwise:
        lines += [
            "# Pairwise comparisons: isolating architecture from precision",
            "",
            "The per-verifier tables above each compare a verifier against its own",
            "English baseline. The two comparisons below instead compare two",
            "conditions directly against each other, with a threshold fit on the",
            "first condition's English scores. This answers how much precision",
            "alone matters, and how much architecture alone matters with precision",
            "held fixed, which the per-verifier tables cannot show on their own.",
            "",
        ]
        for label_a, path_a, label_b, path_b, comparison_label in pairwise:
            analyze_pairwise(label_a, path_a, label_b, path_b, comparison_label, lines)

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "architecture.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
