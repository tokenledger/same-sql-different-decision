# Final numbers: reproducing published statistics that existed in no script

Four numbers cited in the paper that were previously computed ad hoc and not checked into any script. Canonical protocol throughout: 163 databases shuffled with `np.random.default_rng(0)`, the first 81 (floor(163/2)) assigned to calibration and the remaining 82 to test; threshold = the most permissive value whose selective risk on calibration-split English scores is within a 10% target, evaluated at distinct score values only (`xsql.metrics.threshold_at_risk`).

## 1. Pivot vs. quantile transport: paired database bootstrap

Section 7.3 reports pivot's flip-rate advantage over the best-performing score transport (quantile) as significant under a paired database bootstrap. Recomputed here: for each verifier, the per-language flip rate against the English decision (canonical test split) under `pivot` (data/mitigation) and under `quantile` (fit on calibration, applied to test -- scripts/13_transport.py's method) is averaged over the six non-English languages; the same resampled test databases are then used to evaluate both methods within each bootstrap draw, so the difference is paired.

| verifier | pivot mean flip | quantile mean flip | pivot - quantile | 95% CI |
|---|---|---|---|---|
| Llama-3.1-8B | 12.2% | 15.2% | -3.0 pp | [-4.3, -1.6] |
| Qwen2.5-7B | 17.6% | 20.5% | -2.9 pp | [-4.7, -1.1] |

## 2. Per-language quantile-rule coverage spread, mean over ten splits

Section 7.1 reports the per-language quantile rule's coverage spread (max-min coverage over all seven languages; English is left unmapped) averaged over ten independent database splits (seeds 0-9). Each seed refits the threshold AND the quantile map on that seed's own calibration half and measures coverage (and realized risk, to check the accompanying claim that risk control is not preserved) on that seed's own test half.

| verifier | mean spread | std (population) | max realized risk, any lang/seed | seeds 0-9 |
|---|---|---|---|---|
| Llama-3.1-8B | 0.049 | 0.011 | 0.136 | 0.049, 0.058, 0.055, 0.036, 0.031, 0.053, 0.055, 0.067, 0.038, 0.043 |
| Qwen2.5-7B | 0.047 | 0.023 | 0.143 | 0.090, 0.030, 0.077, 0.035, 0.051, 0.025, 0.029, 0.031, 0.070, 0.032 |

## 3. Cross-model English-only decision flips under a shared threshold

Section 8 states that quantizing the same checkpoint, and swapping dense for MoE, change English decisions even with language held fixed. Prespecified 300-question, 39-database screening subset (data/architecture/), canonical seed-0 split, threshold fit on condition A's English calibration scores and applied unchanged to both conditions' English test scores.

| comparison | A | B | English flip rate | threshold |
|---|---|---|---|---|
| bf16 vs 4-bit, same checkpoint | bf16 | 4-bit (nf4) | 8.4% | 0.9526 |
| dense vs MoE, precision-matched (bf16) | dense (4B) | sparse-MoE (30B-A3B) | 20.6% | 0.9526 |

## 4. Aya Expanse 8B saturation, both definitions, and English coverage

Section 8 previously mixed two different saturation statistics in one parenthetical. This reports both cleanly, plus the English coverage the prespecified gate check is actually about.

English AUROC (all 1,500 candidates): 0.6281
Threshold (fit on English calibration scores, 10% target): 1.0000

| definition | value |
|---|---|
| loose: fraction of scores > 0.999, pooled over all seven languages | 93.4% |
| prespecified gate: fraction of English scores within 1e-6 of 0 or 1 | 72.5% |
| English coverage at its own 10%-target threshold (test split) | 19.1% |
| realized English risk at that threshold (test split) | 0.038 |

The paper's saturation gate uses the prespecified (English-only, within-1e-6) definition, 72.5%; the pooled >0.999 figure (93.4%) is a looser statistic mixed in by mistake in an earlier draft and should not be quoted as the gate criterion.

## 5. Canonical operating point: thresholds, AUROC gaps, risk intervals

Threshold fit on calibration-split English at the 10% target (canonical split), then applied to every language on the test split; the full-corpus block refits the threshold on every English score in the corpus (descriptive, in-sample). Risk CIs are question-clustered percentile bootstraps (4,000 resamples).

### Llama-3.1-8B

- canonical threshold: 0.7549130857919153; full-corpus threshold: 0.7057871475061018
- English scores > 0.99: 6.1% (all), 6.6% (calibration split)

| lang | test AUROC | gap vs en | test risk [95% CI] | test cov | full risk [95% CI] | full cov |
|---|---|---|---|---|---|---|
| en | 0.7346 | +0.0000 | 0.078 [0.054, 0.103] | 0.544 | 0.100 [0.082, 0.119] | 0.609 |
| de | 0.7099 | +0.0248 | 0.102 [0.075, 0.130] | 0.579 | 0.109 [0.090, 0.128] | 0.650 |
| es | 0.7318 | +0.0029 | 0.096 [0.072, 0.122] | 0.646 | 0.109 [0.091, 0.128] | 0.696 |
| fr | 0.7140 | +0.0206 | 0.068 [0.043, 0.095] | 0.423 | 0.080 [0.061, 0.098] | 0.492 |
| ja | 0.7035 | +0.0311 | 0.085 [0.061, 0.110] | 0.566 | 0.110 [0.091, 0.129] | 0.653 |
| vi | 0.6866 | +0.0480 | 0.107 [0.081, 0.135] | 0.595 | 0.118 [0.099, 0.138] | 0.664 |
| zh | 0.7256 | +0.0091 | 0.092 [0.067, 0.118] | 0.565 | 0.103 [0.085, 0.121] | 0.639 |

### Qwen2.5-7B

- canonical threshold: 0.9999970976890127; full-corpus threshold: 0.9999832985861913
- English scores > 0.99: 82.3% (all), 83.2% (calibration split)

| lang | test AUROC | gap vs en | test risk [95% CI] | test cov | full risk [95% CI] | full cov |
|---|---|---|---|---|---|---|
| en | 0.7540 | +0.0000 | 0.085 [0.062, 0.109] | 0.610 | 0.099 [0.082, 0.118] | 0.698 |
| de | 0.7158 | +0.0382 | 0.082 [0.056, 0.110] | 0.499 | 0.093 [0.074, 0.112] | 0.603 |
| es | 0.7427 | +0.0113 | 0.074 [0.050, 0.099] | 0.537 | 0.094 [0.077, 0.113] | 0.639 |
| fr | 0.7286 | +0.0253 | 0.079 [0.054, 0.104] | 0.529 | 0.093 [0.076, 0.111] | 0.658 |
| ja | 0.7005 | +0.0535 | 0.094 [0.069, 0.121] | 0.554 | 0.099 [0.082, 0.118] | 0.644 |
| vi | 0.7057 | +0.0483 | 0.087 [0.059, 0.118] | 0.462 | 0.093 [0.075, 0.111] | 0.560 |
| zh | 0.7174 | +0.0366 | 0.089 [0.065, 0.116] | 0.564 | 0.098 [0.081, 0.116] | 0.627 |

