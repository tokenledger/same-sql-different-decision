"""The paper's main figure: risk vs. coverage, per language, per grader.

Each language's risk-coverage curve traces what the verifier could achieve at
any threshold; the curves overlap, because selection quality is similar
everywhere. The marker on each curve is where the single English-calibrated
threshold lands. Those markers sit at very different coverages at similar
risk: the selector still knows which SQL is safer, but a raw English cutoff
invokes it at very different rates.

Curves are drawn in a recessive gray because their overlap is the point;
identity is carried by the labelled markers.

Usage: uv run python scripts/11_risk_coverage_figure.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xsql.config import DATA_DIR, RESULTS_DIR
from xsql.metrics import risk_at_threshold, risk_coverage, threshold_at_risk

BIG = DATA_DIR / "big"
VERIFIERS = [("llama8b", "Llama-3.1-8B"), ("qwen7b", "Qwen2.5-7B")]
LANGS = ["en", "de", "es", "fr", "ja", "vi", "zh"]
PIVOT = "en"
TARGET_RISK = 0.10
SEED = 0

# Okabe-Ito palette. Every marker is direct-labelled, so low contrast between
# neighbouring hues does not affect legibility.
COLOR = {
    "en": "#333333",
    "de": "#E69F00",
    "es": "#56B4E9",
    "fr": "#D55E00",
    "ja": "#009E73",
    "vi": "#CC79A7",
    "zh": "#0072B2",
}
INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"


def load(verifier: str):
    rows = [json.loads(l) for l in (BIG / f"scores_big-{verifier}.jsonl").open()]
    by = defaultdict(dict)
    for r in rows:
        by[r["lang"]][r["candidate_id"]] = r
    ids = sorted(by[PIVOT])
    labels = np.array([by[PIVOT][c]["correct"] for c in ids], dtype=float)
    dbs = np.array([by[PIVOT][c]["db_id"] for c in ids])
    conf = {l: np.array([by[l][c]["confidence"] for c in ids]) for l in LANGS}
    return labels, dbs, conf


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)

    for ax, (verifier, pretty) in zip(axes, VERIFIERS):
        labels, dbs, conf = load(verifier)

        # Fit the threshold on English over half the databases, evaluate on the
        # held-out half, so the operating point is not fitted on what it is shown.
        uniq = sorted(set(dbs))
        perm = np.random.default_rng(SEED).permutation(uniq)
        fit = set(perm[: len(uniq) // 2])
        m_fit = np.array([d in fit for d in dbs])
        m_test = ~m_fit
        thr = threshold_at_risk(labels[m_fit], conf[PIVOT][m_fit], TARGET_RISK)

        lab_t = labels[m_test]
        for lang in LANGS:
            cov, risk = risk_coverage(lab_t, conf[lang][m_test])
            keep = cov >= 0.05  # the far-left tail is a handful of items and is noise
            ax.plot(cov[keep], risk[keep], color=MUTED, lw=1.0, alpha=0.35, zorder=1)

        ax.axhline(
            TARGET_RISK, color=INK, lw=1.0, ls=(0, (4, 3)), alpha=0.55, zorder=2
        )
        ax.text(
            0.985, TARGET_RISK + 0.004, f"{TARGET_RISK:.0%} target",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=INK, alpha=0.7,
        )

        pts = []
        for lang in LANGS:
            r, c = risk_at_threshold(lab_t, conf[lang][m_test], thr)
            pts.append((c, r, lang))
            ax.scatter(
                c, r, s=68, color=COLOR[lang], edgecolor="white", linewidth=1.4,
                zorder=4, clip_on=False,
            )

        # Direct-label every marker. Markers cluster tightly in coverage, so
        # push each label to a free vertical slot instead of a fixed offset.
        placed: list[tuple[float, float]] = []
        for c, r, lang in sorted(pts):
            dy = 11.0
            while any(abs(c - pc) < 0.045 and abs(dy - pdy) < 11 for pc, pdy in placed):
                dy += 11.0
            placed.append((c, dy))
            ax.annotate(
                lang, (c, r), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=9.5, fontweight="bold", color=COLOR[lang],
                zorder=5,
            )

        # Horizontal span annotation, parked below every marker so it never
        # collides with one.
        covs = [p[0] for p in pts]
        y_span = min(r for _, r, _ in pts) - 0.018
        ax.annotate(
            "", xy=(min(covs), y_span), xytext=(max(covs), y_span),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1, alpha=0.55),
            zorder=3,
        )
        ax.text(
            (min(covs) + max(covs)) / 2, y_span - 0.009,
            f"{max(covs)-min(covs):.0%} coverage gap",
            ha="center", va="top", fontsize=8.5, color=INK, alpha=0.8, zorder=5,
        )

        ax.set_title(pretty, fontsize=11.5, color=INK, pad=22, loc="left")
        ax.set_xlabel("Coverage — share of queries executed automatically", fontsize=9.5, color=MUTED)
        ax.set_xlim(0.05, 0.85)
        ax.set_ylim(0.02, 0.20)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    axes[0].set_ylabel("Risk — error rate among executed queries", fontsize=9.5, color=MUTED)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    fig.suptitle(
        "Same SQL, same selection quality — very different automation rates",
        fontsize=13.5, color=INK, x=0.007, ha="left", y=0.985, fontweight="600",
    )
    fig.text(
        0.007, 0.905,
        "Gray curves: each language's achievable risk–coverage trade-off (they overlap). "
        "Markers: where one English-calibrated threshold lands.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS_DIR / f"risk_coverage.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {RESULTS_DIR}/risk_coverage.png and .pdf")


if __name__ == "__main__":
    main()
