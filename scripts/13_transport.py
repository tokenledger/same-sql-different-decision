"""Can a target-language score be mapped onto the English scale?

The decision-level result says an English-calibrated threshold changes 14-27% of
individual execution decisions. This asks whether any score transport restores
the English decisions, not merely equal coverage, which rank matching already
achieves while still flipping which specific queries run.

Five transports, all fitted on a database-disjoint calibration split and none
requiring correctness labels in the target language:

  raw        English threshold applied unchanged
  offset     additive shift on the logit (match mean)            unpaired
  affine     scale + shift on the logit (match mean and sd)      unpaired
  quantile   rank matching: target quantile -> English quantile  unpaired
  isotonic   monotone fit of target score -> its paired English score

Only `isotonic` uses the paired structure: the same SQL scored in both
languages. That extra requirement is only worth it if it beats the unpaired
methods on decision consistency.

Reported per language: decision flips vs. English, unsafe promotions, lost
automation, accepted-set overlap (Jaccard), coverage, and accepted risk.

Usage: uv run python scripts/13_transport.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.isotonic import IsotonicRegression

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import threshold_at_risk

BIG = DATA_DIR / "big"
VERIFIERS = [("llama8b", "Llama-3.1-8B"), ("qwen7b", "Qwen2.5-7B")]
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
# Qwen's scores sit within 1e-6 of 1.0; clipping at 1e-6 merges thousands of
# genuinely distinct values and destroys their ordering. 1e-12 leaves them intact.
EPS = 1e-12
SEED = 0

METHODS = ["raw", "offset", "affine", "quantile", "isotonic"]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load(verifier: str):
    rows = [json.loads(l) for l in (BIG / f"scores_big-{verifier}.jsonl").open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["lang"]][r["candidate_id"]] = r
    ids = sorted(by[PIVOT])
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    conf = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in [PIVOT] + LANGS}
    return labels, dbs, conf


def fit_apply(method, s_cal_lang, s_cal_en, s_test_lang):
    """Map target-language scores onto the English scale. No labels used."""
    if method == "raw":
        return s_test_lang

    if method == "offset":
        z = logit(s_test_lang) + (logit(s_cal_en).mean() - logit(s_cal_lang).mean())
        return sigmoid(z)

    if method == "affine":
        zl, ze = logit(s_cal_lang), logit(s_cal_en)
        scale = ze.std() / zl.std() if zl.std() > 0 else 1.0
        z = (logit(s_test_lang) - zl.mean()) * scale + ze.mean()
        return sigmoid(z)

    if method == "quantile":
        # Unpaired: match the marginal distributions, ignoring which candidate
        # is which. Equivalent to a per-language rank rule.
        q = np.searchsorted(np.sort(s_cal_lang), s_test_lang, side="left") / len(s_cal_lang)
        return np.quantile(s_cal_en, np.clip(q, 0, 1))

    if method == "isotonic":
        # Paired: fit target -> its own English counterpart, monotone so ranking
        # within the language is preserved.
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(s_cal_lang, s_cal_en)
        return iso.predict(s_test_lang)

    raise ValueError(method)


def main() -> None:
    lines = [
        "# Cross-lingual score transport: can the English decisions be recovered?",
        "",
        "All transports fitted on a database-disjoint calibration split, none using",
        "correctness labels in the target language. `flips` is the share of identical",
        "SQL whose execute/defer decision differs from the English decision, the",
        "quantity a transport has to reduce to be worth anything. Coverage parity",
        "alone does not imply decision agreement.",
        "",
    ]

    summary = defaultdict(list)

    for verifier, pretty in VERIFIERS:
        labels, dbs, conf = load(verifier)
        uniq = sorted(set(dbs))
        perm = np.random.default_rng(SEED).permutation(uniq)
        cal = set(perm[: len(uniq) // 2])
        m_cal = np.array([d in cal for d in dbs])
        m_test = ~m_cal

        thr = threshold_at_risk(labels[m_cal], conf[PIVOT][m_cal], TARGET_RISK)
        lab = labels[m_test]
        acc_en = conf[PIVOT][m_test] >= thr
        n_corr, n_incorr = int(lab.sum()), int((1 - lab).sum())

        lines += [
            f"## {pretty}",
            "",
            f"Threshold {thr:.4f}. Held-out: {len(lab)} candidates "
            f"({n_corr} correct / {n_incorr} incorrect). "
            f"English coverage {acc_en.mean():.3f}, risk {1 - lab[acc_en].mean():.3f}.",
            "",
            "| method | lang | flips | unsafe promoted | lost automation | overlap | coverage | risk |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for method in METHODS:
            agg = defaultdict(list)
            for l in LANGS:
                mapped = fit_apply(method, conf[l][m_cal], conf[PIVOT][m_cal], conf[l][m_test])
                acc = mapped >= thr

                flips = float((acc != acc_en).mean())
                promo = int((~acc_en & acc & (lab == 0)).sum())
                demo = int((acc_en & ~acc & (lab == 1)).sum())
                inter = float((acc & acc_en).sum())
                union = float((acc | acc_en).sum())
                jac = inter / union if union else float("nan")
                cov = float(acc.mean())
                risk = float(1 - lab[acc].mean()) if acc.any() else float("nan")

                for k, v in [("flips", flips), ("promo", promo / n_incorr),
                             ("demo", demo / n_corr), ("jac", jac),
                             ("cov", cov), ("risk", risk)]:
                    agg[k].append(v)

                lines.append(
                    f"| {method} | {l} | {flips:.1%} | {promo/n_incorr:.1%} | "
                    f"{demo/n_corr:.1%} | {jac:.3f} | {cov:.3f} | {risk:.3f} |"
                )
            lines.append(
                f"| **{method}** | **mean** | **{np.mean(agg['flips']):.1%}** | "
                f"**{np.mean(agg['promo']):.1%}** | **{np.mean(agg['demo']):.1%}** | "
                f"**{np.mean(agg['jac']):.3f}** | "
                f"**{np.mean(agg['cov']):.3f}** | **{np.mean(agg['risk']):.3f}** |"
            )
            summary[verifier].append(
                (method, np.mean(agg["flips"]), np.mean(agg["promo"]),
                 np.mean(agg["demo"]), np.mean(agg["jac"]),
                 max(agg["cov"]) - min(agg["cov"]), np.mean(agg["risk"]))
            )
        lines.append("")

    lines += ["## Summary: mean over the six non-English languages", "",
              "| verifier | method | flips | unsafe | lost | overlap | coverage spread | risk |",
              "|---|---|---|---|---|---|---|---|"]
    for verifier, _ in VERIFIERS:
        for m, f, p, d, j, cs, r in summary[verifier]:
            lines.append(
                f"| {verifier} | {m} | {f:.1%} | {p:.1%} | {d:.1%} | {j:.3f} | {cs:.3f} | {r:.3f} |"
            )
    lines += [
        "",
        "A transport earns a place in the paper only if it cuts `flips` and `unsafe`",
        "without materially raising `risk`. Shrinking `coverage spread` alone is not",
        "sufficient: a rank rule equalises how many queries run while still changing",
        "which ones do.",
        "",
    ]

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "transport.md").write_text(text)
    print(text[text.index("## Summary"):])
    print(f"full table written to {RESULTS_DIR}/transport.md")


if __name__ == "__main__":
    main()
