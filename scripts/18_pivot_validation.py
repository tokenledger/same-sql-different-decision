"""Is pivot's improvement real, or is it repairing a benchmark defect?

MultiSpider's translations do not always preserve quoted literals (see
scripts/17_translation_audit.py): Vietnamese renders "Pass" as 'Đạt'. The pivot
condition machine-translates that question back into English, and MT restores
37.7% of Vietnamese's lost literals. So part of pivot's apparent benefit could be
repairing the benchmark rather than reducing genuine verifier inconsistency.

The test: recompute pivot's improvement on the literal-clean subset, the items
whose English literal survived the benchmark translation, where there is nothing
for MT to repair. If the improvement holds there, it is not defect repair.

Also reports per-language rates and paired database-bootstrap intervals, since
the paper's own argument is that six-language means hide language-specific harm.

Usage: uv run python scripts/18_pivot_validation.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import threshold_at_risk

BIG, MIT = DATA_DIR / "big", DATA_DIR / "mitigation"
VERIFIERS = [("llama8b", "Llama-3.1-8B"), ("qwen7b", "Qwen2.5-7B")]
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
N_BOOT = 2000
SEED = 0

LIT = re.compile(r"\"([^\"]{2,})\"|“([^”]{2,})”|'([^']{2,})'")


def literals(s):
    return [x for t in LIT.findall(s) for x in t if x]


def defective_items(items):
    """Items whose English literal did not survive the benchmark translation."""
    bad = defaultdict(set)
    for idx, it in items.items():
        el = literals(it["questions"][PIVOT])
        if not el:
            continue
        for l in LANGS:
            if not all(x.lower() in it["questions"][l].lower() for x in el):
                bad[l].add(idx)
    return bad


def main() -> None:
    items = {j["idx"]: j for j in (json.loads(l) for l in (BIG / "items.jsonl").open())}
    bad = defective_items(items)

    lines = [
        "# Is the pivot improvement genuine, or benchmark repair?",
        "",
        "`clean` restricts to items whose English literal survived MultiSpider's own",
        "translation, so machine translation has no defect to repair there. If pivot's",
        "improvement holds on `clean`, it is not an artifact of fixing the benchmark.",
        "",
    ]

    for verifier, pretty in VERIFIERS:
        base = [json.loads(l) for l in (BIG / f"scores_big-{verifier}.jsonl").open()]
        by = defaultdict(dict)
        for r in base:
            by[r["lang"]][r["candidate_id"]] = r
        ids = sorted(by[PIVOT])
        lab = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
        dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
        item_of = np.array([by[PIVOT][c]["item_idx"] for c in ids])
        conf_en = np.array([by[PIVOT][c]["confidence"] for c in ids])
        conf_nat = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in LANGS}

        pv = defaultdict(dict)
        for r in (json.loads(l) for l in (MIT / f"scores_mitigation_pivot-{verifier}.jsonl").open()):
            pv[r["lang"]][r["candidate_id"]] = r["confidence"]
        conf_piv = {l: np.array([pv[l][c] for c in ids]) for l in LANGS}

        uniq = sorted(set(dbs))
        cal = set(np.random.default_rng(SEED).permutation(uniq)[: len(uniq) // 2])
        m = np.array([d in cal for d in dbs])
        t = ~m
        thr = threshold_at_risk(lab[m], conf_en[m], TARGET_RISK)

        a_en = conf_en[t] >= thr
        lab_t, item_t, dbs_t = lab[t], item_of[t], dbs[t]
        n_c, n_i = int(lab_t.sum()), int((1 - lab_t).sum())

        uniq_t = np.unique(dbs_t)
        idx_of = {d: np.flatnonzero(dbs_t == d) for d in uniq_t}
        rng = np.random.default_rng(SEED)
        draws = [
            np.concatenate([idx_of[d] for d in rng.choice(uniq_t, len(uniq_t), replace=True)])
            for _ in range(N_BOOT)
        ]

        lines += [
            f"## {pretty}",
            "",
            f"Held-out {len(lab_t)} candidates ({n_c} correct / {n_i} incorrect), "
            f"threshold {thr:.4f}.",
            "",
            "| lang | flip native | flip pivot | Δ flip [95% CI] | unsafe native | unsafe pivot | Δ flip, literal-clean | n dropped |",
            "|---|---|---|---|---|---|---|---|",
        ]

        worst_nat_flip = worst_piv_flip = 0.0
        worst_nat_uns = worst_piv_uns = 0.0
        improved = 0

        for l in LANGS:
            a_n = conf_nat[l][t] >= thr
            a_p = conf_piv[l][t] >= thr
            f_n = float((a_n != a_en).mean())
            f_p = float((a_p != a_en).mean())
            u_n = float(((~a_en) & a_n & (lab_t == 0)).sum() / n_i)
            u_p = float(((~a_en) & a_p & (lab_t == 0)).sum() / n_i)

            vals = [
                float((a_p[i] != a_en[i]).mean() - (a_n[i] != a_en[i]).mean()) for i in draws
            ]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            sig = "" if lo <= 0 <= hi else " *"

            keep = np.array([i not in bad[l] for i in item_t])
            f_n_c = float((a_n[keep] != a_en[keep]).mean())
            f_p_c = float((a_p[keep] != a_en[keep]).mean())

            improved += f_p < f_n
            worst_nat_flip = max(worst_nat_flip, f_n)
            worst_piv_flip = max(worst_piv_flip, f_p)
            worst_nat_uns = max(worst_nat_uns, u_n)
            worst_piv_uns = max(worst_piv_uns, u_p)

            lines.append(
                f"| {l} | {f_n:.1%} | {f_p:.1%} | {f_p-f_n:+.1%} [{lo:+.1%}, {hi:+.1%}]{sig} "
                f"| {u_n:.1%} | {u_p:.1%} | {f_p_c-f_n_c:+.1%} | {int((~keep).sum())} |"
            )

        lines += [
            "",
            f"- Languages improved: **{improved}/6**",
            f"- Worst per-language flip: {worst_nat_flip:.1%} → **{worst_piv_flip:.1%}**",
            f"- Worst per-language unsafe promotion: {worst_nat_uns:.1%} → **{worst_piv_uns:.1%}**",
            "",
        ]

    lines += [
        "A `Δ flip, literal-clean` close to the all-items `Δ flip` means the",
        "improvement is not explained by machine translation repairing values that",
        "MultiSpider's own translation had altered.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pivot_validation.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
