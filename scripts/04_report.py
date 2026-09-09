"""Slice report.

Two questions:
  1. Does the verifier rank and calibrate equally well in every language?
  2. Does an English-fitted abstention threshold hold its target risk when
     reused on non-English questions?

Writes results/slice_report.md.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from xsql.config import LANGUAGES, PIVOT, RESULTS_DIR, SLICE_DIR
from xsql.data import read_jsonl
from xsql.metrics import (
    auroc,
    brier,
    coverage_at_risk,
    ece,
    risk_at_threshold,
    threshold_at_risk,
)

TARGET_RISKS = [0.10, 0.20, 0.30]


def fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def main(backend: str = "api") -> None:
    scores = read_jsonl(SLICE_DIR / f"scores_{backend}.jsonl")

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in scores:
        by_lang[r["lang"]].append(r)

    # Take the language set from the data, not config, so reports work on runs
    # that used a different set (the Colab run scores 7).
    langs = [l for l in LANGUAGES if l in by_lang]
    langs += sorted(set(by_lang) - set(langs))
    # Align candidate order across languages so differences are truly paired.
    for lang in by_lang:
        by_lang[lang].sort(key=lambda r: r["candidate_id"])

    # Keep only candidates scored successfully in every language, so each
    # language is measured on an identical candidate set.
    scored_everywhere = {
        c
        for c in {r["candidate_id"] for r in scores}
        if all(
            any(r["candidate_id"] == c and r["confidence"] is not None for r in by_lang[lang])
            for lang in langs
        )
    }
    dropped = len(by_lang[PIVOT]) - len(scored_everywhere)
    for lang in by_lang:
        by_lang[lang] = [r for r in by_lang[lang] if r["candidate_id"] in scored_everywhere]

    labels = np.array([r["correct"] for r in by_lang[PIVOT]], dtype=float)
    conf = {lang: np.array([r["confidence"] for r in rs]) for lang, rs in by_lang.items()}

    lines: list[str] = [f"# Cross-lingual verifier slice report ({backend})", ""]
    lines.append(
        f"{len(labels)} candidates, {int(labels.sum())} correct "
        f"({labels.mean():.1%} execution accuracy)"
        + (f"; {dropped} dropped for unparseable confidence" if dropped else "")
        + f". All candidates generated from the {PIVOT} question; only the "
        "question language shown to the verifier varies."
    )
    lines += ["", "## Ranking and calibration by question language", ""]
    lines.append("| lang | AUROC | Brier | ECE | mean conf |")
    lines.append("|---|---|---|---|---|")
    for lang in langs:
        c = conf[lang]
        lines.append(
            f"| {lang} | {fmt(auroc(labels, c))} | {fmt(brier(labels, c))} | "
            f"{fmt(ece(labels, c))} | {fmt(float(c.mean()))} |"
        )

    lines += ["", "## Paired score shift vs. English (same SQL, same label)", ""]
    lines.append("| lang | mean Δconf | mean \\|Δconf\\| | flipped rank order |")
    lines.append("|---|---|---|---|")
    for lang in langs:
        if lang == PIVOT:
            continue
        delta = conf[lang] - conf[PIVOT]
        # How often the language change alone moves a candidate across 0.5.
        flipped = np.mean((conf[lang] >= 0.5) != (conf[PIVOT] >= 0.5))
        lines.append(
            f"| {lang} | {fmt(float(delta.mean()))} | {fmt(float(np.abs(delta).mean()))} "
            f"| {flipped:.1%} |"
        )

    lines += ["", "## Transferring the English abstention threshold", ""]
    lines.append(
        "Threshold fitted on English to hit the target risk, then applied "
        "unchanged to each language's scores."
    )
    lines.append("")
    lines.append("| target risk | lang | threshold | realized risk | coverage | own-language coverage |")
    lines.append("|---|---|---|---|---|---|")
    for target in TARGET_RISKS:
        thr = threshold_at_risk(labels, conf[PIVOT], target)
        for lang in langs:
            if thr is None:
                lines.append(f"| {target:.0%} | {lang} | n/a | n/a | n/a | n/a |")
                continue
            realized, coverage = risk_at_threshold(labels, conf[lang], thr)
            own = coverage_at_risk(labels, conf[lang], target)
            lines.append(
                f"| {target:.0%} | {lang} | {fmt(thr)} | {fmt(realized)} | "
                f"{fmt(coverage)} | {fmt(own)} |"
            )

    report = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"slice_report_{backend}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "api")
