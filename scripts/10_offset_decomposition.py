"""Is the language effect a class-INDEPENDENT score offset?

If a language shifts correct and incorrect SQL by the same amount, the ranking
is preserved and only the scale moves -- which is exactly the pattern that lets
AUROC stay flat while a fixed threshold changes coverage. If instead the shift
differs by class, the language is also changing discrimination, and calling it
"an offset" would be wrong.

For each language l and grader g, on candidate i with label y_i:

    delta(i,l)   = s(i,l) - s(i,en)
    d_correct    = mean of delta over y_i = 1
    d_incorrect  = mean of delta over y_i = 0
    kappa        = d_correct - d_incorrect      <- the interaction

kappa near zero supports (but does not prove) a class-independent offset. This
is an empirical description, not a causal mechanism.

Resampling is over DATABASES: candidates sharing a database share a schema and
difficulty, and the same SQL is scored in every language, so neither judgments
nor questions are independent units here.

Usage: uv run python scripts/10_offset_decomposition.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from xsql.config import DATA_DIR, RESULTS_DIR

BIG = DATA_DIR / "big"
VERIFIERS = ["llama8b", "qwen7b"]
LANGS = ["de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
N_BOOT = 4000
SEED = 0


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


def main() -> None:
    lines = [
        "# Is the language effect a class-independent offset?",
        "",
        "`delta` is the confidence change vs. English on the *same* SQL with the",
        "*same* execution-derived label. If `kappa` (the correct-minus-incorrect",
        "difference) is near zero, the language moves the score scale without",
        "changing discrimination -- which is what allows AUROC to stay flat while a",
        "fixed threshold shifts coverage. Bootstrap resamples DATABASES.",
        "",
    ]

    for verifier in VERIFIERS:
        labels, dbs, conf = load(verifier)
        uniq = np.unique(dbs)
        idx_of = {d: np.flatnonzero(dbs == d) for d in uniq}
        rng = np.random.default_rng(SEED)

        # Pre-draw one set of database resamples, shared across languages so the
        # per-language numbers move together and stay comparable.
        draws = [rng.choice(uniq, len(uniq), replace=True) for _ in range(N_BOOT)]
        draw_idx = [np.concatenate([idx_of[d] for d in pick]) for pick in draws]

        lines += [
            f"## {verifier}",
            "",
            f"{len(labels)} candidates over {len(uniq)} databases "
            f"({int(labels.sum())} correct, {int((1-labels).sum())} incorrect).",
            "",
            "| lang | shift on correct | shift on incorrect | kappa (difference) | 95% CI | class-independent? |",
            "|---|---|---|---|---|---|",
        ]

        for l in LANGS:
            d = conf[l] - conf[PIVOT]
            dc = d[labels == 1].mean()
            di = d[labels == 0].mean()
            kappa = dc - di

            ks = []
            for i in draw_idx:
                dl, ll = d[i], labels[i]
                if ll.sum() == 0 or (1 - ll).sum() == 0:
                    continue
                ks.append(dl[ll == 1].mean() - dl[ll == 0].mean())
            lo, hi = np.percentile(ks, [2.5, 97.5])
            # "Consistent with an offset" means the interaction is both
            # non-significant AND small relative to the shift itself.
            ns = lo <= 0 <= hi
            small = abs(kappa) < 0.5 * max(abs(dc), abs(di), 1e-9)
            verdict = "yes" if (ns or small) else "**no**"
            lines.append(
                f"| {l} | {dc:+.4f} | {di:+.4f} | {kappa:+.4f} | "
                f"[{lo:+.4f}, {hi:+.4f}] | {verdict} |"
            )
        lines.append("")

    lines += [
        "## Reading this",
        "",
        "A small `kappa` supports describing the language effect as an approximately",
        "additive, class-independent score offset **for these graders on this task**.",
        "It is an empirical description, not a causal explanation, and it does not",
        "rule out language-by-item interaction that averages out across items.",
        "",
    ]

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "offset_decomposition.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
