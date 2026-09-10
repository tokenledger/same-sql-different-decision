"""What the score drift does to actual execution decisions.

Coverage and risk are aggregates. This is the per-decision view, which is what
distinguishes a SQL-verification paper from a general evaluator-bias paper: the
verifier's score does not rate text, it decides whether a database action runs.

For one English-calibrated threshold applied unchanged to every language, on
identical SQL with identical execution-derived labels, every candidate falls
into one of three cells:

    demoted    accepted in English, rejected in this language
    promoted   rejected in English, accepted in this language
    agree      same decision either way

Split by the true label, those become the two failure modes that matter:

    lost automation   correct SQL demoted: work needlessly sent to a human
    unsafe execution  incorrect SQL promoted: a wrong query now runs

Zhou et al. (2607.14480) report the unsafe direction for general evaluators in
lower-resource languages. Reporting both directions shows which failure this
application suffers.

Usage: uv run python scripts/12_decision_analysis.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import beta

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import threshold_at_risk

BIG = DATA_DIR / "big"
VERIFIERS = [("llama8b", "Llama-3.1-8B"), ("qwen7b", "Qwen2.5-7B")]
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
N_BOOT = 2000
SEED = 0


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval; unlike a bootstrap it stays informative at k=0."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def load(verifier: str):
    rows = [json.loads(l) for l in (BIG / f"scores_big-{verifier}.jsonl").open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["lang"]][r["candidate_id"]] = r
    ids = sorted(by[PIVOT])
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    item_idx = np.array([by[PIVOT][c]["item_idx"] for c in ids])
    conf = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in [PIVOT] + LANGS}
    return labels, dbs, item_idx, conf


def main() -> None:
    lines = [
        "# What the language shift does to individual execution decisions",
        "",
        "One English-calibrated threshold, applied unchanged. Same SQL, same",
        "database, same execution-derived label; only the question's language",
        "differs, so every flip below is caused by language alone.",
        "",
    ]

    for verifier, pretty in VERIFIERS:
        labels, dbs, item_idx, conf = load(verifier)
        uniq = sorted(set(dbs))
        perm = np.random.default_rng(SEED).permutation(uniq)
        fit = set(perm[: len(uniq) // 2])
        m_fit = np.array([d in fit for d in dbs])
        m_test = ~m_fit
        thr = threshold_at_risk(labels[m_fit], conf[PIVOT][m_fit], TARGET_RISK)

        lab = labels[m_test]
        acc = {l: conf[l][m_test] >= thr for l in [PIVOT] + LANGS}
        items_t = item_idx[m_test]
        uniq_t = np.unique(items_t)
        idx_of = {q: np.flatnonzero(items_t == q) for q in uniq_t}
        rng = np.random.default_rng(SEED)
        draws = [
            np.concatenate([idx_of[d] for d in rng.choice(uniq_t, len(uniq_t), replace=True)])
            for _ in range(N_BOOT)
        ]

        n_corr, n_incorr = int(lab.sum()), int((1 - lab).sum())
        lines += [
            f"## {pretty}",
            "",
            f"Threshold {thr:.4f} (fit on English, {len(fit)} databases). "
            f"Held-out: {len(lab)} candidates, {n_corr} correct / {n_incorr} incorrect.",
            "",
            "| lang | decisions flipped | lost automation | unsafe promotion (of all incorrect) | unsafe promotion (of English-withheld incorrect) | 95% CI, question-clustered |",
            "|---|---|---|---|---|---|",
        ]

        for l in LANGS:
            flip = acc[PIVOT] != acc[l]
            demoted_corr = (acc[PIVOT] & ~acc[l] & (lab == 1)).sum()
            promoted_incorr = (~acc[PIVOT] & acc[l] & (lab == 0)).sum()

            # Question-cluster bootstrap: the five candidates from one question
            # share a database and difficulty, so candidate-level exact-binomial
            # intervals understate the width. Clopper-Pearson is used only for
            # zero-event cells, where every resample also contains zero and a
            # percentile interval collapses to [0, 0]; there it is reported as a
            # candidate-level descriptive bound, not a clustered population bound.
            if promoted_incorr == 0:
                _, hi_cp = clopper_pearson(promoted_incorr, n_incorr)
                ci = (f"0 of {n_incorr} candidate-level opportunities; "
                      f"exact-binomial upper bound {hi_cp:.1%} (unclustered)")
            else:
                vals = []
                for i in draws:
                    den = (lab[i] == 0).sum()
                    if den:
                        vals.append(
                            ((~acc[PIVOT][i]) & acc[l][i] & (lab[i] == 0)).sum() / den
                        )
                lo_b, hi_b = np.percentile(vals, [2.5, 97.5])
                ci = f"[{lo_b:.1%}, {hi_b:.1%}]"

            # Conditional form: the share of incorrect queries withheld in English
            # that this language executes.
            den_cond = (acc[PIVOT] == False).__and__(lab == 0).sum()
            cond = promoted_incorr / den_cond if den_cond else float("nan")

            lines.append(
                f"| {l} | {flip.mean():.1%} | "
                f"{demoted_corr}/{n_corr} = {demoted_corr/n_corr:.1%} | "
                f"{promoted_incorr}/{n_incorr} = {promoted_incorr/n_incorr:.1%} | "
                f"{cond:.1%} | {ci} |"
            )

        # Full transition decomposition. Lost-automation and unsafe-promotion
        # rates have different denominators, so quoting them alone does not show
        # whether they cancel. The accepted-error and accepted-correct deltas
        # explain why aggregate selective risk barely moves.
        lines += [
            "",
            "**Transition decomposition.** Counts of held-out candidates by true label and",
            "by (English decision -> target-language decision). `d errors` and `d correct`",
            "are the net change in the accepted set's numerator and denominator, which is",
            "what determines whether aggregate risk moves.",
            "",
            "| lang | correct: exec->defer | correct: defer->exec | incorrect: exec->defer | incorrect: defer->exec | d errors | d correct | risk (en) | risk (lang) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for l in LANGS:
            cd = int((acc[PIVOT] & ~acc[l] & (lab == 1)).sum())
            cp = int((~acc[PIVOT] & acc[l] & (lab == 1)).sum())
            idm = int((acc[PIVOT] & ~acc[l] & (lab == 0)).sum())
            ip = int((~acc[PIVOT] & acc[l] & (lab == 0)).sum())
            d_err, d_cor = ip - idm, cp - cd
            r_en = 1 - lab[acc[PIVOT]].mean() if acc[PIVOT].any() else float("nan")
            r_l = 1 - lab[acc[l]].mean() if acc[l].any() else float("nan")
            lines.append(
                f"| {l} | {cd} | {cp} | {idm} | {ip} | {d_err:+d} | {d_cor:+d} "
                f"| {r_en:.3f} | {r_l:.3f} |"
            )
        lines.append("")

    lines += [
        "## Reading this",
        "",
        "`unsafe execution` is the rate at which a wrong query, correctly withheld",
        "in English, is executed automatically once the same request arrives in",
        "another language. `lost automation` is the mirror image: a correct query",
        "needlessly escalated to a human. Both are caused by language alone, since",
        "the SQL and its correctness label are held fixed.",
        "",
    ]

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "decision_analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
