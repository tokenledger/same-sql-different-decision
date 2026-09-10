"""Is one verifier more cross-lingually sensitive than another?

Raw disagreement between two verifiers conflates two things: they have different
English policies to begin with, and they may react differently to language. A
dense and an MoE verifier may disagree in English, and that says nothing about
cross-lingual stability.

So each model is measured against its own English decisions:

    I_m(lang) = P(decision_{m,lang} != decision_{m,en})

and compare models on that:

    dI(lang) = I_B(lang) - I_A(lang)

which cancels the English-baseline disagreement entirely.

Two threshold regimes, because a model that accepts almost nothing trivially
looks stable:

  risk      each model's own English threshold at the 10% risk target
  covmatch  each model's English threshold set to a common English coverage,
            so a saturated model cannot win by abstaining

CIs are a paired database bootstrap: both models scored the identical candidate
set, so the same databases are resampled for both at once and the difference is
taken within each draw.

Usage: uv run python scripts/15_within_model_sensitivity.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import threshold_at_risk

ARCH = DATA_DIR / "architecture"
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
N_BOOT = 2000
SEED = 0

# (label, file stem) pairs. Each row asks: does this intervention change how
# sensitive the verifier is to the question's language?
COMPARISONS = [
    ("dense vs sparse-MoE (bf16, same family)",
     "yesno-qwen3-4b-bf16", "yesno-qwen3-30ba3b-moe-bf16"),
    ("bf16 vs 4-bit (same checkpoint)",
     "yesno-qwen3-4b-bf16", "yesno-qwen3-4b-nf4"),
    ("Yes/No vs CORRECT/INCORRECT (same model)",
     "yesno-qwen7b", "correctincorrect-qwen7b"),
]


def load(stem: str):
    rows = [json.loads(l) for l in (ARCH / f"scores_architecture_{stem}.jsonl").open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["lang"]][r["candidate_id"]] = r
    ids = sorted(by[PIVOT])
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    conf = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in [PIVOT] + LANGS}
    return ids, labels, dbs, conf


def decisions(conf, mask, thr):
    return {l: conf[l][mask] >= thr for l in [PIVOT] + LANGS}


def instability(dec, lab):
    """Per-language flip / unsafe-promotion / lost-automation rates vs own English."""
    out = {}
    n_c, n_i = (lab == 1).sum(), (lab == 0).sum()
    for l in LANGS:
        flip = float((dec[l] != dec[PIVOT]).mean())
        unsafe = float((~dec[PIVOT] & dec[l] & (lab == 0)).sum() / n_i) if n_i else np.nan
        lost = float((dec[PIVOT] & ~dec[l] & (lab == 1)).sum() / n_c) if n_c else np.nan
        out[l] = (flip, unsafe, lost)
    return out


def main() -> None:
    lines = [
        "# Within-model cross-lingual sensitivity",
        "",
        "Each model is compared against **its own** English decisions, so baseline",
        "disagreement between models cancels. `dI` is the second model's instability",
        "minus the first's: negative means the second model is less language-sensitive.",
        "",
        "Two regimes. `risk`: each model's own English threshold at the 10% risk",
        "target. `covmatch`: thresholds set to a common English coverage, so a",
        "saturated model cannot appear stable merely by abstaining.",
        "",
    ]

    for title, stem_a, stem_b in COMPARISONS:
        ids_a, lab, dbs, conf_a = load(stem_a)
        ids_b, lab_b, dbs_b, conf_b = load(stem_b)
        # The two files must cover the same candidates in the same order or the
        # pairing is silently wrong.
        assert ids_a == ids_b and np.array_equal(lab, lab_b), f"{stem_a}/{stem_b} misaligned"

        uniq = sorted(set(dbs))
        perm = np.random.default_rng(SEED).permutation(uniq)
        cal = set(perm[: len(uniq) // 2])
        m_cal = np.array([d in cal for d in dbs])
        m_t = ~m_cal
        lab_t = lab[m_t]

        thr_a = threshold_at_risk(lab[m_cal], conf_a[PIVOT][m_cal], TARGET_RISK)
        thr_b = threshold_at_risk(lab[m_cal], conf_b[PIVOT][m_cal], TARGET_RISK)

        lines += [f"## {title}", "", f"A = `{stem_a}`  ·  B = `{stem_b}`", ""]
        if thr_a is None or thr_b is None:
            lines += ["No feasible English threshold for one model at this risk target.", ""]
            continue

        # Common English coverage: the smaller of the two risk-regime coverages,
        # so both models can reach it.
        cov_a = float((conf_a[PIVOT][m_t] >= thr_a).mean())
        cov_b = float((conf_b[PIVOT][m_t] >= thr_b).mean())
        target_cov = min(cov_a, cov_b)
        q = 1 - target_cov
        thr_a_cm = float(np.quantile(conf_a[PIVOT][m_t], q))
        thr_b_cm = float(np.quantile(conf_b[PIVOT][m_t], q))

        for regime, ta, tb in [("risk", thr_a, thr_b), ("covmatch", thr_a_cm, thr_b_cm)]:
            dec_a = decisions(conf_a, m_t, ta)
            dec_b = decisions(conf_b, m_t, tb)
            ia, ib = instability(dec_a, lab_t), instability(dec_b, lab_t)

            ca = float(dec_a[PIVOT].mean())
            cb = float(dec_b[PIVOT].mean())
            lines += [
                f"### regime: {regime}  (English coverage A={ca:.3f}, B={cb:.3f})",
                "",
                "| lang | flip A | flip B | dI flip [95% CI] | unsafe A | unsafe B | d unsafe | lost A | lost B | d lost |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]

            # Paired database bootstrap: identical candidates, so resample
            # databases once per draw and evaluate both models on that draw.
            dbs_t = dbs[m_t]
            uniq_t = np.unique(dbs_t)
            idx_of = {d: np.flatnonzero(dbs_t == d) for d in uniq_t}
            rng = np.random.default_rng(SEED)
            draws = [
                np.concatenate([idx_of[d] for d in rng.choice(uniq_t, len(uniq_t), replace=True)])
                for _ in range(N_BOOT)
            ]

            for l in LANGS:
                d_flip = ib[l][0] - ia[l][0]
                vals = []
                for i in draws:
                    fa = (dec_a[l][i] != dec_a[PIVOT][i]).mean()
                    fb = (dec_b[l][i] != dec_b[PIVOT][i]).mean()
                    vals.append(fb - fa)
                lo, hi = np.percentile(vals, [2.5, 97.5])
                sig = "" if lo <= 0 <= hi else " *"
                lines.append(
                    f"| {l} | {ia[l][0]:.1%} | {ib[l][0]:.1%} | "
                    f"{d_flip:+.1%} [{lo:+.1%}, {hi:+.1%}]{sig} | "
                    f"{ia[l][1]:.1%} | {ib[l][1]:.1%} | {ib[l][1]-ia[l][1]:+.1%} | "
                    f"{ia[l][2]:.1%} | {ib[l][2]:.1%} | {ib[l][2]-ia[l][2]:+.1%} |"
                )

            ma = np.mean([ia[l][0] for l in LANGS])
            mb = np.mean([ib[l][0] for l in LANGS])
            mvals = []
            for i in draws:
                fa = np.mean([(dec_a[l][i] != dec_a[PIVOT][i]).mean() for l in LANGS])
                fb = np.mean([(dec_b[l][i] != dec_b[PIVOT][i]).mean() for l in LANGS])
                mvals.append(fb - fa)
            mlo, mhi = np.percentile(mvals, [2.5, 97.5])
            lines += [
                f"| **mean** | **{ma:.1%}** | **{mb:.1%}** | "
                f"**{mb-ma:+.1%} [{mlo:+.1%}, {mhi:+.1%}]**"
                f"{'' if mlo <= 0 <= mhi else ' *'} | | | | | | |",
                "",
            ]
        lines.append("")

    lines += [
        "`*` marks a difference whose 95% CI excludes zero.",
        "",
        "Read `dI` only in the `covmatch` regime when either model saturates: at its",
        "own risk threshold a model that accepts almost nothing shows few flips",
        "because it is barely deciding anything, not because it is language-robust.",
        "",
    ]

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "within_model_sensitivity.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
