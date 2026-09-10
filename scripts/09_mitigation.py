"""Does showing the verifier English alongside the native question close the gap?

Four conditions on the same fixed candidate SQL and execution-derived labels:
  en         English question only                       (baseline, from the big run)
  native     target-language question only               (the failure mode, from the big run)
  bilingual  native and English together in one prompt   (needs no translator)
  pivot      machine translation of the native question  (needs MT)

The headline number is coverage spread: the gap between the best- and
worst-served language at a single abstention threshold fitted on English. That
is what fails to transfer, so it is what a mitigation has to shrink.

Usage: uv run python scripts/09_mitigation.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import auroc, risk_at_threshold, threshold_at_risk

BIG = DATA_DIR / "big"
MIT = DATA_DIR / "mitigation"
VERIFIERS = ["llama8b", "qwen7b"]
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
N_BOOT = 2000
SEED = 0


def load(verifier: str):
    """conf[condition][lang] -> array aligned to a shared candidate ordering."""
    base = [json.loads(l) for l in (BIG / f"scores_big-{verifier}.jsonl").open()]
    by = defaultdict(dict)
    for r in base:
        by[r["lang"]][r["candidate_id"]] = r

    ids = sorted(by[PIVOT])
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    items = np.array([by[PIVOT][c]["item_idx"] for c in ids])

    conf = {"en": {PIVOT: np.array([by[PIVOT][c]["confidence"] for c in ids])}}
    conf["native"] = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in LANGS}

    for cond in ("bilingual", "pivot"):
        rows = [
            json.loads(l)
            for l in (MIT / f"scores_mitigation_{cond}-{verifier}.jsonl").open()
        ]
        d = defaultdict(dict)
        for r in rows:
            d[r["lang"]][r["candidate_id"]] = r["confidence"]
        # A missing candidate here would silently misalign the arrays, so fail loudly.
        for l in LANGS:
            missing = [c for c in ids if c not in d[l]]
            if missing:
                raise ValueError(f"{cond}/{verifier}/{l}: {len(missing)} candidates missing")
        conf[cond] = {l: np.array([d[l][c] for c in ids]) for l in LANGS}

    return labels, dbs, items, conf


def coverage_by_lang(labels, conf, cond, thr, mask):
    """Coverage per language under `cond`, with English carried from its own condition."""
    out = {PIVOT: risk_at_threshold(labels[mask], conf["en"][PIVOT][mask], thr)[1]}
    for l in LANGS:
        out[l] = risk_at_threshold(labels[mask], conf[cond][l][mask], thr)[1]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    lines = ["# Mitigation: does showing English alongside the native question help?", ""]

    for verifier in VERIFIERS:
        labels, dbs, items, conf = load(verifier)
        uniq_db = sorted(set(dbs))
        perm = np.random.default_rng(SEED).permutation(uniq_db)
        fit = set(perm[: len(uniq_db) // 2])
        m_fit = np.array([d in fit for d in dbs])
        m_test = ~m_fit

        # Threshold fitted on English over the FIT databases, evaluated on held-out ones.
        thr = threshold_at_risk(labels[m_fit], conf["en"][PIVOT][m_fit], TARGET_RISK)

        lines += [
            f"## {verifier}",
            "",
            f"{len(labels)} candidates, {int(labels.sum())} correct. English "
            f"{TARGET_RISK:.0%}-risk threshold fitted on {len(fit)} databases "
            f"= {thr:.4f}, evaluated on the {len(uniq_db)-len(fit)} held out.",
            "",
            "### Coverage per language, by condition",
            "",
            "| condition | " + " | ".join([PIVOT] + LANGS) + " | spread |",
            "|---" * (len(LANGS) + 3) + "|",
        ]

        spreads = {}
        for cond in ("native", "bilingual", "pivot"):
            cov = coverage_by_lang(labels, conf, cond, thr, m_test)
            vals = [cov[l] for l in [PIVOT] + LANGS]
            spread = max(vals) - min(vals)
            spreads[cond] = spread
            lines.append(
                f"| {cond} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | **{spread:.3f}** |"
            )

        # Bootstrap the spread, resampling questions.
        test_items = items[m_test]
        uniq_items = np.unique(test_items)
        idx_of = {i: np.flatnonzero(test_items == i) for i in uniq_items}
        lab_t = labels[m_test]
        conf_t = {c: {l: v[m_test] for l, v in d.items()} for c, d in conf.items()}

        lines += ["", "### Coverage spread, bootstrapped over questions", "",
                  "| condition | spread | 95% CI | vs native |", "|---|---|---|---|"]
        boots = {}
        for cond in ("native", "bilingual", "pivot"):
            vals = []
            for _ in range(N_BOOT):
                pick = rng.choice(uniq_items, len(uniq_items), replace=True)
                i = np.concatenate([idx_of[k] for k in pick])
                cov = [risk_at_threshold(lab_t[i], conf_t["en"][PIVOT][i], thr)[1]]
                cov += [risk_at_threshold(lab_t[i], conf_t[cond][l][i], thr)[1] for l in LANGS]
                vals.append(max(cov) - min(cov))
            boots[cond] = np.array(vals)
            lo, hi = np.percentile(boots[cond], [2.5, 97.5])
            delta = ""
            if cond != "native":
                d = boots[cond] - boots["native"]
                dlo, dhi = np.percentile(d, [2.5, 97.5])
                sig = "**better**" if dhi < 0 else ("worse" if dlo > 0 else "no change")
                delta = f"{d.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}] {sig}"
            lines.append(f"| {cond} | {spreads[cond]:.3f} | [{lo:.3f}, {hi:.3f}] | {delta} |")

        # Does the mitigation cost discrimination? AUROC should not collapse.
        lines += ["", "### AUROC by condition (held-out databases)", "",
                  "| condition | " + " | ".join(LANGS) + " |", "|---" * (len(LANGS) + 1) + "|"]
        for cond in ("native", "bilingual", "pivot"):
            row = []
            for l in LANGS:
                a = auroc(lab_t, conf_t[cond][l])
                row.append("n/a" if a is None else f"{a:.3f}")
            lines.append(f"| {cond} | " + " | ".join(row) + " |")
        a_en = auroc(lab_t, conf_t["en"][PIVOT])
        lines += ["", f"English reference AUROC: {a_en:.3f}", ""]

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "mitigation.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
