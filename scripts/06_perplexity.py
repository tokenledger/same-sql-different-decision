"""Does question perplexity explain the cross-language confidence shift?

Mechanism under test: the verifier is less confident when the question text is
less familiar to it. If so, per-question Δlog-perplexity (language vs English)
should predict per-candidate Δconfidence.

Perplexity is measured on the question alone, under the verifier model itself,
so it reflects that model's familiarity rather than a generic fluency score.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xsql.config import LOCAL_VERIFIER_MODEL, PIVOT, RESULTS_DIR, SLICE_DIR

N_ITEMS = 150  # subsample: 8B on MPS is ~1s per forward pass
SEED = 0


def question_logppl(model, tok, text: str) -> float:
    """Mean negative log-likelihood per token of `text`."""
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    if ids.shape[1] < 2:
        return float("nan")
    with torch.inference_mode():
        out = model(ids, labels=ids)
    return float(out.loss)  # already mean NLL over tokens


def main(backend: str) -> None:
    items = [json.loads(l) for l in (SLICE_DIR / "items.jsonl").open()]
    scores = [json.loads(l) for l in (SLICE_DIR / f"scores_{backend}.jsonl").open()]

    langs = sorted({r["lang"] for r in scores})
    rng = np.random.default_rng(SEED)
    sample = [items[i] for i in rng.choice(len(items), min(N_ITEMS, len(items)), replace=False)]

    tok = AutoTokenizer.from_pretrained(LOCAL_VERIFIER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_VERIFIER_MODEL, dtype=torch.bfloat16, device_map="mps"
    ).eval()

    ppl: dict[tuple[str, str], float] = {}
    for n, it in enumerate(sample, 1):
        for lang in langs:
            ppl[(it["idx"], lang)] = question_logppl(model, tok, it["questions"][lang])
        if n % 25 == 0:
            print(f"  {n}/{len(sample)} items", flush=True)

    # Per-candidate confidence, keyed for pairing.
    conf: dict[str, dict[str, float]] = defaultdict(dict)
    item_of: dict[str, str] = {}
    for r in scores:
        conf[r["candidate_id"]][r["lang"]] = r["confidence"]
        item_of[r["candidate_id"]] = r["item_idx"]

    keep = {it["idx"] for it in sample}
    cids = [c for c in conf if item_of[c] in keep and len(conf[c]) == len(langs)]

    lines = [
        f"# Question perplexity vs. confidence shift ({backend})",
        "",
        f"Verifier: `{LOCAL_VERIFIER_MODEL}`. {len(sample)} questions, {len(cids)} candidates.",
        "Perplexity is mean NLL per token of the question text under the verifier.",
        "",
        "| lang | mean logPPL | Δ vs en | mean Δconf |",
        "|---|---|---|---|",
    ]
    mean_ppl = {
        l: float(np.nanmean([ppl[(it["idx"], l)] for it in sample])) for l in langs
    }
    mean_dconf = {
        l: float(np.mean([conf[c][l] - conf[c][PIVOT] for c in cids])) for l in langs
    }
    for l in sorted(langs, key=lambda x: (x != PIVOT, x)):
        lines.append(
            f"| {l} | {mean_ppl[l]:.3f} | {mean_ppl[l] - mean_ppl[PIVOT]:+.3f} "
            f"| {mean_dconf[l]:+.4f} |"
        )

    # Across-language correlation (6 points, descriptive only).
    others = [l for l in langs if l != PIVOT]
    x = np.array([mean_ppl[l] - mean_ppl[PIVOT] for l in others])
    y = np.array([mean_dconf[l] for l in others])
    r_lang = float(np.corrcoef(x, y)[0, 1]) if len(others) > 2 else float("nan")

    # Per-candidate correlation, pooled across languages. This uses
    # within-language variation rather than 6 aggregate points.
    dp, dc = [], []
    for c in cids:
        for l in others:
            p = ppl[(item_of[c], l)] - ppl[(item_of[c], PIVOT)]
            if np.isfinite(p):
                dp.append(p)
                dc.append(conf[c][l] - conf[c][PIVOT])
    r_cand = float(np.corrcoef(dp, dc)[0, 1])

    lines += [
        "",
        f"- Correlation across the {len(others)} language means: **r = {r_lang:+.3f}** "
        "(6 points; descriptive only)",
        f"- Correlation across {len(dp)} individual (candidate, language) pairs: "
        f"**r = {r_cand:+.3f}**",
        "",
        "A strong negative r would mean: the less familiar the question text, the "
        "lower the verifier's confidence in identical SQL.",
    ]

    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"perplexity_{backend}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "big-llama8b")
