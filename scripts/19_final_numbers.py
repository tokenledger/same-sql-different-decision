"""Reproducibility closer: recompute published numbers that existed in no script.

Four statistics cited in the paper were, until now, produced ad hoc (notebook
cells, one-off REPL sessions) rather than checked into a script anyone could
rerun. This recomputes each one directly from data/, using the SAME canonical
protocol as every primary result in this project (scripts/12_decision_analysis.py,
scripts/13_transport.py, scripts/14_architecture.py, scripts/18_pivot_validation.py):
163 databases shuffled with `np.random.default_rng(0)`, the first 81
(floor(163/2)) assigned to calibration and the remaining 82 to test; the
English threshold is the most permissive value whose selective risk on
calibration-split English scores stays within a 10% target, evaluated at
distinct score values only (`xsql.metrics.threshold_at_risk`). Reused directly:
`threshold_at_risk`, `risk_at_threshold`, `auroc` from `xsql.metrics`.

The four numbers:

  1. Pivot vs. quantile-transport paired database bootstrap on decision flips
     (Section 7.3 / sec:pivot). Is pivot's flip-rate reduction over the
     best-performing score transport (quantile, Section 7.2) significant?
  2. Per-language quantile-rule coverage spread, averaged over ten database
     splits (seeds 0-9) rather than reported on the single canonical split
     (Section 7.1 / sec:coverage).
  3. Cross-model English-only decision flips under a shared threshold: how
     much do quantization (bf16 vs 4-bit) and architecture (dense vs MoE)
     alone move decisions with language held fixed at English (Section 8 /
     sec:robust), from data/architecture/.
  4. Aya Expanse 8B saturation under both the loose definition (>0.999,
     pooled over all seven languages) and the prespecified gate definition
     (within 1e-6 of {0,1}, English only), plus its English coverage
     (Section 8 / sec:robust).

Usage: uv run python scripts/19_final_numbers.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import auroc, risk_at_threshold, threshold_at_risk

BIG = DATA_DIR / "big"
MIT = DATA_DIR / "mitigation"
ARCH = DATA_DIR / "architecture"
VERIFIERS = [("llama8b", "Llama-3.1-8B"), ("qwen7b", "Qwen2.5-7B")]
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
N_BOOT = 2000
SEED = 0
SAT_EPS = 1e-6


def quantile_transport(s_cal_lang: np.ndarray, s_cal_en: np.ndarray, s_eval_lang: np.ndarray) -> np.ndarray:
    """Unpaired rank matching: target-language quantile -> English quantile.

    Identical to the `quantile` method in scripts/13_transport.py.
    """
    q = np.searchsorted(np.sort(s_cal_lang), s_eval_lang, side="left") / len(s_cal_lang)
    return np.quantile(s_cal_en, np.clip(q, 0, 1))


def canonical_split(dbs: np.ndarray, seed: int = SEED) -> tuple[set[str], set[str]]:
    """163 databases, seed-shuffled, first floor(N/2) to calibration, the rest
    to test -- the split used everywhere else in this project."""
    uniq = sorted(set(dbs.tolist()))
    perm = np.random.default_rng(seed).permutation(uniq)
    n_cal = len(uniq) // 2
    return set(perm[:n_cal].tolist()), set(perm[n_cal:].tolist())


def load_big(verifier: str):
    rows = [json.loads(l) for l in (BIG / f"scores_big-{verifier}.jsonl").open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["lang"]][r["candidate_id"]] = r
    ids = sorted(by[PIVOT])
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    conf = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in [PIVOT] + LANGS}
    return ids, labels, dbs, conf


def load_pivot(verifier: str, ids: list[str]):
    pv = defaultdict(dict)
    for r in (json.loads(l) for l in (MIT / f"scores_mitigation_pivot-{verifier}.jsonl").open()):
        pv[r["lang"]][r["candidate_id"]] = r["confidence"]
    return {l: np.array([pv[l][c] for c in ids]) for l in LANGS}


def load_arch(stem: str):
    rows = [json.loads(l) for l in (ARCH / f"scores_architecture_{stem}.jsonl").open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["lang"]][r["candidate_id"]] = r
    ids = sorted(by[PIVOT])
    present = [l for l in [PIVOT] + LANGS if l in by]
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    conf = {l: np.array([by[l][c]["confidence"] for c in ids], dtype=float) for l in present}
    return labels, dbs, conf, present


# ---------------------------------------------------------------------------
# 1. Pivot vs. quantile transport, paired database bootstrap
# ---------------------------------------------------------------------------


def pivot_vs_quantile() -> list[str]:
    lines = [
        "## 1. Pivot vs. quantile transport: paired database bootstrap",
        "",
        "Section 7.3 reports pivot's flip-rate advantage over the best-performing "
        "score transport (quantile) as significant under a paired database "
        "bootstrap. Recomputed here: for each verifier, the per-language flip "
        "rate against the English decision (canonical test split) under "
        "`pivot` (data/mitigation) and under `quantile` (fit on calibration, "
        "applied to test -- scripts/13_transport.py's method) is averaged over "
        "the six non-English languages; the same resampled test databases are "
        "then used to evaluate both methods within each bootstrap draw, so the "
        "difference is paired.",
        "",
        "| verifier | pivot mean flip | quantile mean flip | pivot - quantile | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for verifier, pretty in VERIFIERS:
        ids, labels, dbs, conf = load_big(verifier)
        conf_piv = load_pivot(verifier, ids)
        cal, _test = canonical_split(dbs)
        m_cal = np.array([d in cal for d in dbs])
        m_test = ~m_cal
        thr = threshold_at_risk(labels[m_cal], conf[PIVOT][m_cal], TARGET_RISK)

        acc_en = conf[PIVOT][m_test] >= thr
        dbs_t = dbs[m_test]

        flip_piv, flip_quant = {}, {}
        for l in LANGS:
            a_piv = conf_piv[l][m_test] >= thr
            mapped = quantile_transport(conf[l][m_cal], conf[PIVOT][m_cal], conf[l][m_test])
            a_quant = mapped >= thr
            flip_piv[l] = (a_piv != acc_en).astype(float)
            flip_quant[l] = (a_quant != acc_en).astype(float)

        mean_piv = float(np.mean([flip_piv[l].mean() for l in LANGS]))
        mean_quant = float(np.mean([flip_quant[l].mean() for l in LANGS]))

        uniq_t = np.unique(dbs_t)
        idx_of = {d: np.flatnonzero(dbs_t == d) for d in uniq_t}
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(N_BOOT):
            sampled = rng.choice(uniq_t, len(uniq_t), replace=True)
            idx = np.concatenate([idx_of[d] for d in sampled])
            mp = np.mean([flip_piv[l][idx].mean() for l in LANGS])
            mq = np.mean([flip_quant[l][idx].mean() for l in LANGS])
            diffs.append((mp - mq) * 100)
        lo, hi = np.percentile(diffs, [2.5, 97.5])

        lines.append(
            f"| {pretty} | {mean_piv:.1%} | {mean_quant:.1%} | "
            f"{(mean_piv - mean_quant) * 100:+.1f} pp | [{lo:+.1f}, {hi:+.1f}] |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 2. Per-language quantile-rule coverage spread, averaged over ten splits
# ---------------------------------------------------------------------------


def quantile_spread_over_seeds() -> list[str]:
    lines = [
        "## 2. Per-language quantile-rule coverage spread, mean over ten splits",
        "",
        "Section 7.1 reports the per-language quantile rule's coverage spread "
        "(max-min coverage over all seven languages; English is left unmapped) "
        "averaged over ten independent database splits (seeds 0-9). Each seed "
        "refits the threshold AND the quantile map on that seed's own "
        "calibration half and measures coverage (and realized risk, to check "
        "the accompanying claim that risk control is not preserved) on that "
        "seed's own test half.",
        "",
        "| verifier | mean spread | std (population) | max realized risk, any lang/seed | seeds 0-9 |",
        "|---|---|---|---|---|",
    ]
    for verifier, pretty in VERIFIERS:
        _ids, labels, dbs, conf = load_big(verifier)
        spreads = []
        max_risk = 0.0
        for seed in range(10):
            cal, _test = canonical_split(dbs, seed=seed)
            m_cal = np.array([d in cal for d in dbs])
            m_test = ~m_cal
            thr = threshold_at_risk(labels[m_cal], conf[PIVOT][m_cal], TARGET_RISK)
            if thr is None:
                continue
            risk_en, cov_en = risk_at_threshold(labels[m_test], conf[PIVOT][m_test], thr)
            covs = {PIVOT: cov_en}
            max_risk = max(max_risk, risk_en)
            for l in LANGS:
                mapped = quantile_transport(conf[l][m_cal], conf[PIVOT][m_cal], conf[l][m_test])
                risk_l, cov_l = risk_at_threshold(labels[m_test], mapped, thr)
                covs[l] = cov_l
                max_risk = max(max_risk, risk_l)
            spreads.append(max(covs.values()) - min(covs.values()))
        arr = np.array(spreads)
        lines.append(
            f"| {pretty} | {arr.mean():.3f} | {arr.std():.3f} | {max_risk:.3f} | "
            + ", ".join(f"{s:.3f}" for s in arr) + " |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 3. Cross-model English-only decision flips under a shared threshold
# ---------------------------------------------------------------------------


def english_flip(stem_a: str, stem_b: str) -> tuple[float, float]:
    """Threshold fit on A's English calibration scores, applied UNCHANGED to
    both A's and B's English test scores. Same methodology as
    `analyze_pairwise` in scripts/14_architecture.py, restricted to English."""
    labels, dbs, conf_a, _ = load_arch(stem_a)
    _, _, conf_b, _ = load_arch(stem_b)
    cal, _test = canonical_split(dbs)
    m_cal = np.array([d in cal for d in dbs])
    m_test = ~m_cal
    thr = threshold_at_risk(labels[m_cal], conf_a[PIVOT][m_cal], TARGET_RISK)
    acc_a = conf_a[PIVOT][m_test] >= thr
    acc_b = conf_b[PIVOT][m_test] >= thr
    return float((acc_a != acc_b).mean()), thr


def cross_model_flips() -> list[str]:
    lines = [
        "## 3. Cross-model English-only decision flips under a shared threshold",
        "",
        "Section 8 states that quantizing the same checkpoint, and swapping "
        "dense for MoE, change English decisions even with language held "
        "fixed. Prespecified 300-question, 39-database screening subset "
        "(data/architecture/), canonical seed-0 split, threshold fit on "
        "condition A's English calibration scores and applied unchanged to "
        "both conditions' English test scores.",
        "",
        "| comparison | A | B | English flip rate | threshold |",
        "|---|---|---|---|---|",
    ]
    for stem_a, stem_b, label_a, label_b, comparison in [
        ("yesno-qwen3-4b-bf16", "yesno-qwen3-4b-nf4",
         "bf16", "4-bit (nf4)", "bf16 vs 4-bit, same checkpoint"),
        ("yesno-qwen3-4b-bf16", "yesno-qwen3-30ba3b-moe-bf16",
         "dense (4B)", "sparse-MoE (30B-A3B)", "dense vs MoE, precision-matched (bf16)"),
    ]:
        flip, thr = english_flip(stem_a, stem_b)
        lines.append(f"| {comparison} | {label_a} | {label_b} | {flip:.1%} | {thr:.4f} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 4. Aya Expanse 8B saturation, both definitions, and English coverage
# ---------------------------------------------------------------------------


def aya_saturation() -> list[str]:
    lines = [
        "## 4. Aya Expanse 8B saturation, both definitions, and English coverage",
        "",
        "Section 8 previously mixed two different saturation statistics in one "
        "parenthetical. This reports both cleanly, plus the English coverage "
        "the prespecified gate check is actually about.",
        "",
    ]
    labels, dbs, conf, present = load_arch("yesno-aya-expanse-8b")

    en_auroc = auroc(labels, conf[PIVOT])
    all_conf = np.concatenate([conf[l] for l in present])
    loose = float((all_conf > 0.999).mean())
    strict_en = float(
        ((np.abs(conf[PIVOT]) < SAT_EPS) | (np.abs(1 - conf[PIVOT]) < SAT_EPS)).mean()
    )

    cal, _test = canonical_split(dbs)
    m_cal = np.array([d in cal for d in dbs])
    m_test = ~m_cal
    thr = threshold_at_risk(labels[m_cal], conf[PIVOT][m_cal], TARGET_RISK)
    if thr is not None:
        risk_en, cov_en = risk_at_threshold(labels[m_test], conf[PIVOT][m_test], thr)
    else:
        risk_en, cov_en = float("nan"), float("nan")

    lines += [
        f"English AUROC (all {len(labels):,} candidates): {en_auroc:.4f}"
        if en_auroc is not None else "English AUROC: undefined (single class)",
        f"Threshold (fit on English calibration scores, 10% target): "
        f"{'n/a' if thr is None else f'{thr:.4f}'}",
        "",
        "| definition | value |",
        "|---|---|",
        f"| loose: fraction of scores > 0.999, pooled over all seven languages | {loose:.1%} |",
        f"| prespecified gate: fraction of English scores within 1e-6 of 0 or 1 | {strict_en:.1%} |",
        f"| English coverage at its own 10%-target threshold (test split) | {cov_en:.1%} |",
        f"| realized English risk at that threshold (test split) | {risk_en:.3f} |",
        "",
        f"The paper's saturation gate uses the prespecified (English-only, "
        f"within-1e-6) definition, {strict_en:.1%}; the pooled >0.999 figure "
        f"({loose:.1%}) is a looser statistic mixed in by mistake in an earlier "
        f"draft and should not be quoted as the gate criterion.",
        "",
    ]
    return lines


def canonical_operating_point() -> list[str]:
    """Exact canonical thresholds, test AUROC gaps, and question-clustered risk CIs.

    These are quoted in the paper's Section 4/5 text and in the full-data
    appendix table, and previously existed in no results file."""
    from xsql.metrics import auroc, risk_at_threshold

    langs = ["en", "de", "es", "fr", "ja", "vi", "zh"]
    lines = [
        "## 5. Canonical operating point: thresholds, AUROC gaps, risk intervals",
        "",
        "Threshold fit on calibration-split English at the 10% target (canonical split), "
        "then applied to every language on the test split; the full-corpus block refits "
        "the threshold on every English score in the corpus (descriptive, in-sample). Risk CIs are "
        "question-clustered percentile bootstraps (4,000 resamples).",
        "",
    ]
    for stem, pretty in [("big-llama8b", "Llama-3.1-8B"), ("big-qwen7b", "Qwen2.5-7B")]:
        rows = [json.loads(l) for l in (BIG / f"scores_{stem}.jsonl").open()]
        by = defaultdict(dict)
        for r in rows:
            by[r["lang"]][r["candidate_id"]] = r
        ids = sorted(by["en"])
        labels = np.array([by["en"][c]["correct"] for c in ids], dtype=float)
        dbs = np.array([by["en"][c]["db_id"] for c in ids])
        items = np.array([c.split("#")[0] for c in ids])
        conf = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in langs}
        uniq = sorted(set(dbs))
        perm = np.random.default_rng(SEED).permutation(uniq)
        cal = set(perm[: len(uniq) // 2])
        m = np.array([d in cal for d in dbs])
        t = ~m

        def boot_risk(lab, sc, thr, it, n=4000, seed=SEED):
            rng = np.random.default_rng(seed)
            u = np.unique(it)
            idx = {q: np.flatnonzero(it == q) for q in u}
            out = []
            for _ in range(n):
                ii = np.concatenate([idx[q] for q in rng.choice(u, len(u), replace=True)])
                out.append(risk_at_threshold(lab[ii], sc[ii], thr)[0])
            return np.percentile(out, [2.5, 97.5])

        thr = threshold_at_risk(labels[m], conf["en"][m], 0.10)
        thr_full = threshold_at_risk(labels, conf["en"], 0.10)
        a_en = auroc(labels[t], conf["en"][t])
        lines += [
            f"### {pretty}",
            "",
            f"- canonical threshold: {thr!r}; full-corpus threshold: {thr_full!r}",
            f"- English scores > 0.99: {(conf['en'] > 0.99).mean():.1%} (all), "
            f"{(conf['en'][m] > 0.99).mean():.1%} (calibration split)",
            "",
            "| lang | test AUROC | gap vs en | test risk [95% CI] | test cov | full risk [95% CI] | full cov |",
            "|---|---|---|---|---|---|---|",
        ]
        for l in langs:
            a = auroc(labels[t], conf[l][t])
            r, c = risk_at_threshold(labels[t], conf[l][t], thr)
            lo, hi = boot_risk(labels[t], conf[l][t], thr, items[t])
            rf, cf = risk_at_threshold(labels, conf[l], thr_full)
            lof, hif = boot_risk(labels, conf[l], thr_full, items)
            lines.append(
                f"| {l} | {a:.4f} | {a_en - a:+.4f} | {r:.3f} [{lo:.3f}, {hi:.3f}] | {c:.3f} | "
                f"{rf:.3f} [{lof:.3f}, {hif:.3f}] | {cf:.3f} |"
            )
        lines.append("")
    return lines


def worked_example(candidate_id: str = "train:0#0") -> list[str]:
    """Appendix worked example: execution counts, paired scores, and decisions
    for one test-split candidate under the canonical thresholds."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from xsql.execute import execute

    items = {json.loads(l)["idx"]: json.loads(l) for l in (BIG / "items.jsonl").open()}
    cand = next(
        json.loads(l) for l in (BIG / "candidates.jsonl").open()
        if json.loads(l)["candidate_id"] == candidate_id
    )
    it = items[cand["item_idx"]]
    gold_rows = execute(it["db_id"], it["gold_sql"]).rows
    cand_rows = execute(it["db_id"], cand["sql"]).rows
    lines = [
        "## 6. Worked example (appendix)",
        "",
        f"Candidate `{candidate_id}` on `{it['db_id']}`.",
        "",
        f"- English question: {it['questions']['en']}",
        f"- Vietnamese question: {it['questions']['vi']}",
        f"- Gold SQL: `{it['gold_sql']}` -> {gold_rows}",
        f"- Candidate SQL: `{cand['sql']}` -> {cand_rows}",
        f"- Label: {'correct' if cand['correct'] else 'incorrect'}",
        "",
        "| verifier | threshold | " + " | ".join(["en", "de", "es", "fr", "ja", "vi", "zh"]) + " |",
        "|---|---|" + "---|" * 7,
    ]
    for stem, pretty in [("big-llama8b", "Llama-3.1-8B"), ("big-qwen7b", "Qwen2.5-7B")]:
        rows = [json.loads(l) for l in (BIG / f"scores_{stem}.jsonl").open()]
        by = defaultdict(dict)
        for r in rows:
            by[r["lang"]][r["candidate_id"]] = r
        ids = sorted(by["en"])
        labels = np.array([by["en"][c]["correct"] for c in ids], dtype=float)
        dbs = np.array([by["en"][c]["db_id"] for c in ids])
        cal, _ = canonical_split(dbs)
        m = np.array([d in cal for d in dbs])
        conf_en = np.array([by["en"][c]["confidence"] for c in ids])
        thr = threshold_at_risk(labels[m], conf_en[m], TARGET_RISK)
        cells = []
        for l in ["en", "de", "es", "fr", "ja", "vi", "zh"]:
            sc = by[l][candidate_id]["confidence"]
            cells.append(f"{sc:.7f} {'execute' if sc >= thr else 'defer'}")
        lines.append(f"| {pretty} | {thr:.7f} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> None:
    lines = [
        "# Final numbers: reproducing published statistics that existed in no script",
        "",
        "Four numbers cited in the paper that were previously computed ad hoc and "
        "not checked into any script. Canonical protocol throughout: 163 "
        f"databases shuffled with `np.random.default_rng({SEED})`, the first 81 "
        "(floor(163/2)) assigned to calibration and the remaining 82 to test; "
        "threshold = the most permissive value whose selective risk on "
        f"calibration-split English scores is within a {TARGET_RISK:.0%} "
        "target, evaluated at distinct score values only "
        "(`xsql.metrics.threshold_at_risk`).",
        "",
    ]
    lines += pivot_vs_quantile()
    lines += quantile_spread_over_seeds()
    lines += cross_model_flips()
    lines += aya_saturation()
    lines += canonical_operating_point()
    lines += worked_example()

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final_numbers.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
