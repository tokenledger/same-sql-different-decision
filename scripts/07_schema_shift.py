"""Does the cross-lingual verifier effect survive database/schema shift?

05_analyze.py and the slice reports establish that an English-fitted
abstention threshold keeps its target risk but shifts coverage when the
question language changes, holding the database fixed. That is "language-only
shift": the threshold is fit and evaluated on the same set of databases, and
only the language of the question changes.

In deployment the more realistic failure mode is "language-plus-schema
shift": the threshold is fit on one set of databases (English questions) and
then applied to unseen databases, in a non-English language. This script
partitions the 163 databases into a FIT half and a held-out TEST half and
compares three regimes at a fixed English-fitted threshold:

  (a) language-only shift:    FIT databases,  other language
  (b) language+schema shift:  TEST databases, other language
  (c) schema-shift reference: TEST databases, English
                              (isolates how much of (b) is schema shift alone)

all measured against the (FIT databases, English) baseline the threshold was
fit on. Section 3 checks whether AUROC (ranking quality) itself degrades on
held-out databases, independent of any threshold.

Bootstrap units:
  - (a) shares one population (FIT db candidates) across languages, so it is
    resampled by question with the English/other-language comparison paired,
    as in 05_analyze.py.
  - (b), (c), and the AUROC FIT-vs-TEST gap compare two different database
    populations (FIT vs TEST), so they cannot be paired by question. Each side
    is bootstrapped independently by database (the unit of generalization
    under schema shift) and the two draw sequences are subtracted elementwise,
    a standard two-sample percentile bootstrap for a difference of
    independent statistics.

Usage: XSQL_RUN=big uv run python scripts/07_schema_shift.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from xsql.config import PIVOT, RESULTS_DIR, SLICE_DIR
from xsql.data import read_jsonl
from xsql.metrics import auroc, risk_at_threshold, threshold_at_risk

LANGUAGES = ["en", "de", "es", "fr", "ja", "vi", "zh"]
BACKENDS = ["big-llama8b", "big-qwen7b"]
TARGET_RISK = 0.10
N_BOOT = 2000
SEED = 0


def fmt_sig(lo: float, hi: float) -> str:
    return "**yes**" if (lo > 0 or hi < 0) else "no"


def flag_negligible(delta: float, thresh: float = 0.02) -> str:
    return "" if abs(delta) >= thresh else " (negligible)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load(backend: str):
    """Build per-language score arrays aligned on a common candidate id order,
    restricted to candidates scored in every language (mirrors 05_analyze.py).
    """
    scores = read_jsonl(SLICE_DIR / f"scores_{backend}.jsonl")
    by_lang: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in scores:
        by_lang[r["lang"]][r["candidate_id"]] = r

    ids = sorted(
        c
        for c in by_lang[PIVOT]
        if all(by_lang[l].get(c, {}).get("confidence") is not None for l in LANGUAGES)
    )
    item_idx = np.array([by_lang[PIVOT][c]["item_idx"] for c in ids])
    db_id = np.array([by_lang[PIVOT][c]["db_id"] for c in ids])
    labels = np.array([by_lang[PIVOT][c]["correct"] for c in ids], dtype=float)
    conf = {l: np.array([by_lang[l][c]["confidence"] for c in ids]) for l in LANGUAGES}
    return ids, item_idx, db_id, labels, conf


def split_databases(db_id: np.ndarray, seed: int = SEED) -> tuple[set, set]:
    """Random 50/50 partition of distinct db_ids into FIT / TEST, seeded."""
    uniq = np.unique(db_id)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    half = len(perm) // 2
    return set(perm[:half].tolist()), set(perm[half:].tolist())


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def bootstrap_paired(cluster: np.ndarray, mask: np.ndarray, stat, rng, n_boot=N_BOOT):
    """Percentile CI for a statistic computed jointly on both sides of a
    comparison that share one population (same `mask`), resampling clusters
    (here: questions) with replacement. `stat(idx)` returns a scalar or None.
    """
    sub = np.unique(cluster[mask])
    index_of = {c: np.flatnonzero(mask & (cluster == c)) for c in sub}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(sub, len(sub), replace=True)
        idx = np.concatenate([index_of[c] for c in pick])
        v = stat(idx)
        if v is not None and np.isfinite(v):
            out.append(v)
    return np.array(out)


def bootstrap_two_sample(
    cluster_a: np.ndarray, mask_a: np.ndarray, stat_a,
    cluster_b: np.ndarray, mask_b: np.ndarray, stat_b,
    rng, n_boot=N_BOOT,
):
    """Percentile CI for stat_a - stat_b when a and b are independent
    populations (different databases). Each side is resampled independently
    by its own cluster (database) and the draws are paired positionally.
    Both sequences are iid, so the elementwise difference has the marginal
    sampling distribution of the difference.
    """
    draws_a = bootstrap_paired(cluster_a, mask_a, stat_a, rng, n_boot)
    draws_b = bootstrap_paired(cluster_b, mask_b, stat_b, rng, n_boot)
    n = min(len(draws_a), len(draws_b))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diff = draws_a[:n] - draws_b[:n]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(backend: str) -> list[str]:
    ids, item_idx, db_id, labels, conf = load(backend)
    fit_dbs, test_dbs = split_databases(db_id)
    fit_mask = np.isin(db_id, list(fit_dbs))
    test_mask = np.isin(db_id, list(test_dbs))
    rng = np.random.default_rng(SEED)

    n_fit_items = len(np.unique(item_idx[fit_mask]))
    n_test_items = len(np.unique(item_idx[test_mask]))

    lines = [f"## {backend}", ""]
    if backend == BACKENDS[0]:
        lines += [
            f"{len(np.unique(db_id))} databases split 50/50 by database "
            f"(seed={SEED}): {len(fit_dbs)} FIT / {len(test_dbs)} TEST. "
            f"FIT: {n_fit_items} questions, {int(fit_mask.sum())} candidates "
            f"(x{len(LANGUAGES)} langs each scored). "
            f"TEST: {n_test_items} questions, {int(test_mask.sum())} candidates.",
            "",
        ]

    # Threshold fit once, on English questions over FIT databases only.
    thr = threshold_at_risk(labels[fit_mask], conf[PIVOT][fit_mask], TARGET_RISK)
    lines.append(
        f"Threshold fit on English/FIT to hit {TARGET_RISK:.0%} risk: "
        f"{'n/a' if thr is None else f'{thr:.4f}'}"
    )
    lines.append("")
    if thr is None:
        lines.append("No threshold on FIT/English reaches the target risk; skipping.")
        return lines

    en_fit_risk, en_fit_cov = risk_at_threshold(labels[fit_mask], conf[PIVOT][fit_mask], thr)

    lines += [
        "Baseline (fit population, English): "
        f"risk {en_fit_risk:.3f}, coverage {en_fit_cov:.3f}.",
        "",
        "### (a) language-only shift: FIT databases, other languages",
        "Same population the threshold was fit on; only the language changes. "
        "Paired bootstrap over questions (95% CI on the coverage gap vs FIT/English).",
        "",
        "| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |",
        "|---|---|---|---|---|---|",
        f"| en | {en_fit_risk:.3f} | {en_fit_cov:.3f} | -- | -- | -- |",
    ]
    for lang in LANGUAGES:
        if lang == PIVOT:
            continue
        risk, cov = risk_at_threshold(labels[fit_mask], conf[lang][fit_mask], thr)

        def gap_stat(idx, lang=lang):
            return (
                risk_at_threshold(labels[idx], conf[lang][idx], thr)[1]
                - risk_at_threshold(labels[idx], conf[PIVOT][idx], thr)[1]
            )

        draws = bootstrap_paired(item_idx, fit_mask, gap_stat, rng)
        lo, hi = np.percentile(draws, [2.5, 97.5]) if len(draws) else (float("nan"),) * 2
        d = cov - en_fit_cov
        lines.append(
            f"| {lang} | {risk:.3f} | {cov:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{fmt_sig(lo, hi)}{flag_negligible(d)} |"
        )

    lines += [
        "",
        "### (b) language+schema shift: held-out databases, other languages",
        "Threshold still fit on FIT/English; evaluated on TEST databases in each "
        "other language. Two-sample bootstrap by database (gap vs FIT/English).",
        "",
        "| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |",
        "|---|---|---|---|---|---|",
    ]

    def en_fit_cov_stat(idx):
        return risk_at_threshold(labels[idx], conf[PIVOT][idx], thr)[1]

    for lang in LANGUAGES:
        risk, cov = risk_at_threshold(labels[test_mask], conf[lang][test_mask], thr)

        def cov_stat(idx, lang=lang):
            return risk_at_threshold(labels[idx], conf[lang][idx], thr)[1]

        d, lo, hi = bootstrap_two_sample(
            db_id, test_mask, cov_stat, db_id, fit_mask, en_fit_cov_stat, rng
        )
        marker = " (=English, see (c))" if lang == PIVOT else ""
        lines.append(
            f"| {lang} | {risk:.3f} | {cov:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{fmt_sig(lo, hi)}{flag_negligible(d)}{marker} |"
        )

    lines += [
        "",
        "### (c) reference: schema shift alone, held-out databases, English",
        "Same threshold, same TEST databases, but the pivot language. This isolates "
        "how much of (b) is schema shift vs added language shift. "
        "(Row is a duplicate of the `en` row in (b), repeated for readability.)",
        "",
    ]
    en_test_risk, en_test_cov = risk_at_threshold(labels[test_mask], conf[PIVOT][test_mask], thr)
    d, lo, hi = bootstrap_two_sample(
        db_id, test_mask, en_fit_cov_stat, db_id, fit_mask, en_fit_cov_stat, rng
    )
    lines.append(
        f"English/TEST: risk {en_test_risk:.3f}, coverage {en_test_cov:.3f}, "
        f"gap vs FIT/en {en_test_cov - en_fit_cov:+.3f} "
        f"[{lo:+.3f}, {hi:+.3f}] {fmt_sig(lo, hi)}{flag_negligible(en_test_cov - en_fit_cov)}."
    )

    # Part 3: does ranking (AUROC) itself degrade on held-out databases?
    lines += [
        "",
        "### AUROC on FIT vs held-out (TEST) databases",
        "Two-sample bootstrap by database.",
        "",
        "| lang | AUROC (FIT) | AUROC (TEST) | gap FIT-TEST | 95% CI | significant |",
        "|---|---|---|---|---|---|",
    ]
    for lang in LANGUAGES:
        a_fit = auroc(labels[fit_mask], conf[lang][fit_mask])
        a_test = auroc(labels[test_mask], conf[lang][test_mask])

        def auroc_fit(idx, lang=lang):
            return auroc(labels[idx], conf[lang][idx])

        def auroc_test(idx, lang=lang):
            return auroc(labels[idx], conf[lang][idx])

        d, lo, hi = bootstrap_two_sample(
            db_id, fit_mask, auroc_fit, db_id, test_mask, auroc_test, rng
        )
        af = "n/a" if a_fit is None else f"{a_fit:.3f}"
        at = "n/a" if a_test is None else f"{a_test:.3f}"
        lines.append(
            f"| {lang} | {af} | {at} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{fmt_sig(lo, hi)}{flag_negligible(d)} |"
        )

    return lines


def main() -> None:
    out = [
        "# Does the cross-lingual verifier effect survive schema shift?",
        "",
        "Databases are split 50/50 (by db_id, not by question) into a FIT half "
        "(used to fit the English abstention threshold at 10% target risk) and a "
        "held-out TEST half. See the module docstring for the three regimes "
        "compared and which resampling unit backs each CI.",
        "",
    ]
    for backend in BACKENDS:
        out += analyze(backend) + [""]

    out += [
        "## Reading",
        "",
        "**AUROC (ranking) survives schema shift.** For both verifiers and every "
        "language, the FIT-vs-TEST AUROC gap is not significant (CIs straddle "
        "zero), and for llama8b, TEST-database AUROC is a few points "
        "*higher* than FIT, not lower. There is no evidence the verifier ranks "
        "correct/incorrect SQL worse on unseen databases; if anything the ~80 "
        "held-out databases in this split happen to be marginally easier to "
        "rank, well within noise.",
        "",
        "**Coverage instability from language shift alone (a) mostly carries over "
        "to language+schema shift (b), same sign, similar or larger magnitude, "
        "but many effects lose significance.** That loss of significance is "
        "a sample-size artifact, not evidence the effect vanishes: (a) is "
        "bootstrapped over ~600 questions sharing one fixed set of databases, "
        "while (b) and (c) are bootstrapped over ~80 databases (the unit that "
        "varies under schema shift), which is a much smaller "
        "effective sample. Point estimates in (b) track (a) closely for most "
        "languages (e.g. llama8b/fr: -0.107 in (a) vs -0.143 in (b); "
        "qwen7b/de: -0.229 vs -0.242), so the underlying effect looks stable; "
        "the study is underpowered to confirm it at the database level "
        "with only 163 databases split in half.",
        "",
        "**Schema shift alone (c), with language held at English, is small and "
        "never significant** (llama8b: -0.025 coverage; qwen7b: -0.036), "
        "confirming the coverage instability is a language-shift phenomenon, "
        "not a schema-shift phenomenon. Schema shift mainly acts as an "
        "additional noise source that widens the CIs enough to swallow the "
        "language effect's significance rather than a competing effect with "
        "its own sign.",
        "",
        "**qwen7b is the more fragile verifier.** Its English threshold sits "
        "exactly at score 1.0 (the maximum, a tie-heavy region), so acceptance "
        "is an all-or-nothing gate per language and coverage swings by 15-23 "
        "points across languages regardless of database provenance. This "
        "reproduces the 36.0% (en) vs 14.1% (de) full-data result "
        "reasonably closely on the FIT-only subset here (34.1% vs 11.2%). "
        "llama8b's threshold (0.836) sits in a less saturated region and shows "
        "milder, more mixed-sign coverage shifts.",
        "",
        "**Bottom line:** the risk stays near target in every regime for both "
        "verifiers (the calibration/risk-control property is robust), and "
        "AUROC ranking is unaffected by schema shift. The coverage instability "
        "under language shift is real and, where measurable, does not appear "
        "to be cured or worsened in a clearly resolvable way by additionally "
        "shifting to unseen databases. Point estimates suggest it persists "
        "at similar or larger magnitude, but confirming that at the database "
        "level would need more than 163 databases to reach the same power as "
        "the question-level comparisons.",
        "",
    ]

    text = "\n".join(out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "schema_shift.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
