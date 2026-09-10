"""Do the mitigations restore decisions, or only coverage?

scripts/09_mitigation.py evaluates bilingual and pivot prompting by coverage
spread, the aggregate metric this project argues is inadequate. This applies
the decision-level metrics of scripts/12_decision_analysis.py to the same
conditions, so the mitigations are judged by the standard the paper sets.

Reference policy is the English decision at the English-calibrated threshold.
English is the calibrated reference, not ground truth; the execution-derived
label decides whether a change is harmful.

Usage: uv run python scripts/16_mitigation_decisions.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import threshold_at_risk

BIG, MIT = DATA_DIR / "big", DATA_DIR / "mitigation"
VERIFIERS = [("llama8b", "Llama-3.1-8B"), ("qwen7b", "Qwen2.5-7B")]
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
SEED = 0


def load(verifier: str):
    base = [json.loads(l) for l in (BIG / f"scores_big-{verifier}.jsonl").open()]
    by = defaultdict(dict)
    for r in base:
        by[r["lang"]][r["candidate_id"]] = r
    ids = sorted(by[PIVOT])
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])

    conf = {"en": {PIVOT: np.array([by[PIVOT][c]["confidence"] for c in ids])}}
    conf["native"] = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in LANGS}
    for cond in ("bilingual", "pivot"):
        d = defaultdict(dict)
        for r in (json.loads(l) for l in (MIT / f"scores_mitigation_{cond}-{verifier}.jsonl").open()):
            d[r["lang"]][r["candidate_id"]] = r["confidence"]
        for l in LANGS:
            assert all(c in d[l] for c in ids), f"{cond}/{verifier}/{l} incomplete"
        conf[cond] = {l: np.array([d[l][c] for c in ids]) for l in LANGS}
    return ids, labels, dbs, conf


def main() -> None:
    lines = [
        "# Do the mitigations restore decisions, or only coverage?",
        "",
        "Decision-level metrics for the bilingual and pivot conditions, against the",
        "English reference policy at the English-calibrated threshold. Held-out",
        "databases only. `flip` counts identical SQL whose execute/defer decision",
        "differs from English; `overlap` is the Jaccard similarity of the accepted",
        "sets.",
        "",
    ]

    for verifier, pretty in VERIFIERS:
        ids, labels, dbs, conf = load(verifier)
        uniq = sorted(set(dbs))
        perm = np.random.default_rng(SEED).permutation(uniq)
        cal = set(perm[: len(uniq) // 2])
        m_cal = np.array([d in cal for d in dbs])
        t = ~m_cal
        thr = threshold_at_risk(labels[m_cal], conf["en"][PIVOT][m_cal], TARGET_RISK)

        lab = labels[t]
        a_en = conf["en"][PIVOT][t] >= thr
        n_c, n_i = int(lab.sum()), int((1 - lab).sum())

        lines += [
            f"## {pretty}",
            "",
            f"Threshold {thr:.4f}. Held-out {len(lab)} candidates "
            f"({n_c} correct / {n_i} incorrect); English coverage {a_en.mean():.3f}, "
            f"risk {1 - lab[a_en].mean():.3f}.",
            "",
            "| condition | flip | unsafe promoted | lost automation | overlap | coverage | risk |",
            "|---|---|---|---|---|---|---|",
        ]

        for cond in ("native", "bilingual", "pivot"):
            agg = defaultdict(list)
            for l in LANGS:
                a = conf[cond][l][t] >= thr
                agg["flip"].append(float((a != a_en).mean()))
                agg["unsafe"].append(float(((~a_en) & a & (lab == 0)).sum() / n_i))
                agg["lost"].append(float((a_en & ~a & (lab == 1)).sum() / n_c))
                un = float((a | a_en).sum())
                agg["jac"].append(float((a & a_en).sum() / un) if un else np.nan)
                agg["cov"].append(float(a.mean()))
                agg["risk"].append(float(1 - lab[a].mean()) if a.any() else np.nan)
            lines.append(
                f"| {cond} | {np.mean(agg['flip']):.1%} | {np.mean(agg['unsafe']):.1%} | "
                f"{np.mean(agg['lost']):.1%} | {np.mean(agg['jac']):.3f} | "
                f"{np.mean(agg['cov']):.3f} | {np.mean(agg['risk']):.3f} |"
            )

        lines += ["", "Per language, flip rate:", "",
                  "| condition | " + " | ".join(LANGS) + " |", "|---" * 7 + "|"]
        for cond in ("native", "bilingual", "pivot"):
            row = []
            for l in LANGS:
                a = conf[cond][l][t] >= thr
                row.append(f"{float((a != a_en).mean()):.1%}")
            lines.append(f"| {cond} | " + " | ".join(row) + " |")
        lines.append("")

    lines += [
        "A mitigation that lowers coverage spread but leaves `flip` and `overlap`",
        "unchanged has equalised how many queries run without restoring which ones.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "mitigation_decisions.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
