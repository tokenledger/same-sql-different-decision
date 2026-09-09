# Final numbers: reproducing published statistics that existed in no script

Four numbers cited in the paper that were previously computed ad hoc and not checked into any script. Canonical protocol throughout: 163 databases shuffled with `np.random.default_rng(0)`, the first 81 (floor(163/2)) assigned to calibration and the remaining 82 to test; threshold = the most permissive value whose selective risk on calibration-split English scores is within a 10% target, evaluated at distinct score values only (`xsql.metrics.threshold_at_risk`).

## 1. Pivot vs. quantile transport: paired database bootstrap

Section 7.3 reports pivot's flip-rate advantage over the best-performing score transport (quantile) as significant under a paired database bootstrap. Recomputed here: for each verifier, the per-language flip rate against the English decision (canonical test split) under `pivot` (data/mitigation) and under `quantile` (fit on calibration, applied to test -- scripts/13_transport.py's method) is averaged over the six non-English languages; the same resampled test databases are then used to evaluate both methods within each bootstrap draw, so the difference is paired.

| verifier | pivot mean flip | quantile mean flip | pivot - quantile | 95% CI |
|---|---|---|---|---|
| Llama-3.1-8B | 10.5% | 12.8% | -2.3 pp | [-3.7, -0.9] |
| Qwen2.5-7B | 11.9% | 17.3% | -5.4 pp | [-7.0, -3.8] |

## 2. Per-language quantile-rule coverage spread, mean over ten splits

Section 7.1 reports the per-language quantile rule's coverage spread (max-min coverage over all seven languages; English is left unmapped) averaged over ten independent database splits (seeds 0-9). Each seed refits the threshold AND the quantile map on that seed's own calibration half and measures coverage (and realized risk, to check the accompanying claim that risk control is not preserved) on that seed's own test half.

| verifier | mean spread | std (population) | max realized risk, any lang/seed | seeds 0-9 |
|---|---|---|---|---|
| Llama-3.1-8B | 0.036 | 0.010 | 0.157 | 0.038, 0.034, 0.053, 0.035, 0.014, 0.033, 0.039, 0.040, 0.049, 0.029 |
| Qwen2.5-7B | 0.051 | 0.012 | 0.149 | 0.021, 0.061, 0.064, 0.053, 0.041, 0.054, 0.061, 0.048, 0.047, 0.058 |

## 3. Cross-model English-only decision flips under a shared threshold

Section 8 states that quantizing the same checkpoint, and swapping dense for MoE, change English decisions even with language held fixed. Prespecified 300-question, 39-database screening subset (data/architecture/), canonical seed-0 split, threshold fit on condition A's English calibration scores and applied unchanged to both conditions' English test scores.

| comparison | A | B | English flip rate | threshold |
|---|---|---|---|---|
| bf16 vs 4-bit, same checkpoint | bf16 | 4-bit (nf4) | 7.0% | 1.0000 |
| dense vs MoE, precision-matched (bf16) | dense (4B) | sparse-MoE (30B-A3B) | 21.7% | 1.0000 |

## 4. Aya Expanse 8B saturation, both definitions, and English coverage

Section 8 previously mixed two different saturation statistics in one parenthetical. This reports both cleanly, plus the English coverage the prespecified gate check is actually about.

English AUROC (all 1,500 candidates): 0.5998
Threshold (fit on English calibration scores, 10% target): 1.0000

| definition | value |
|---|---|
| loose: fraction of scores > 0.999, pooled over all seven languages | 93.4% |
| prespecified gate: fraction of English scores within 1e-6 of 0 or 1 | 72.5% |
| English coverage at its own 10%-target threshold (test split) | 11.7% |
| realized English risk at that threshold (test split) | 0.094 |

The paper's saturation gate uses the prespecified (English-only, within-1e-6) definition, 72.5%; the pooled >0.999 figure (93.4%) is a looser statistic mixed in by mistake in an earlier draft and should not be quoted as the gate criterion.

