# Robustness to the operating point and to the database split

Same saved scores and labels as the main analysis; no new inference.

## 1. Threshold sensitivity, canonical split (seed 0)

English threshold refit at each risk target on the 81 calibration databases,
then applied unchanged to every language on the 82 test databases. A target
at or above the verifier's unconditional English error rate on the
calibration split is met by accepting nearly everything, so flips vanish
trivially there; English coverage is the column to read first.

### Llama-3.1-8B

| target | threshold | en coverage | en risk | flip range (6 langs) | mean flip | unsafe promotion range (of all incorrect) |
|---|---|---|---|---|---|---|
| 5% | 0.946591 | 24.6% | 0.026 | 9.2%-13.8% | 11.9% | 0.0%-2.7% |
| 10% | 0.754913 | 54.4% | 0.078 | 13.9%-19.5% | 16.0% | 1.6%-16.9% |
| 15% | 0.0675553 | 99.1% | 0.156 | 0.6%-1.5% | 1.0% | 0.2%-2.7% |

### Qwen2.5-7B

| target | threshold | en coverage | en risk | flip range (6 langs) | mean flip | unsafe promotion range (of all incorrect) |
|---|---|---|---|---|---|---|
| 5% | 1 | 16.7% | 0.044 | 11.0%-14.7% | 13.2% | 0.0%-1.9% |
| 10% | 0.999997 | 61.0% | 0.085 | 17.4%-27.2% | 21.6% | 4.1%-10.5% |
| 15% | 1.81897e-09 | 99.0% | 0.158 | 1.2%-2.6% | 2.0% | 1.2%-2.1% |

## 2. Score-precision stress test (canonical split, 10% target)

Decisions recomputed after rounding every stored probability and the fixed
calibrated threshold to float32. `changed` is the share of (candidate,
language) decisions on the test split that differ from the full-precision
policy. This is a storage-precision conversion test, not a new fp32 model run.

| verifier | threshold | float32 threshold | en coverage | en coverage (f32) | decisions changed (7 langs) | distinct en scores | distinct en scores (f32) |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.7549130857919153 | 0.7549130916595459 | 54.4% | 54.4% | 0.0% | 3138 | 2670 |
| Qwen2.5-7B | 0.9999970976890127 | 0.999997079372406 | 61.0% | 61.1% | 0.2% | 1534 | 575 |

## 3. Repeated database splits (seeds 0-9)

Each seed reshuffles the 163 databases; the English threshold is refit at the
10% target on the first 81 and evaluated on the remaining 82. Seed 0 is the
canonical split used in the paper. Ranges are descriptive, not confidence
intervals: the splits overlap and are not independent replications.

### Llama-3.1-8B

| seed | threshold | en coverage | mean flip (6 langs) | worst-lang flip | quantile flip | pivot flip | pivot - quantile | unsafe de | unsafe es | unsafe fr | unsafe ja | unsafe vi | unsafe zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.754913 | 54.4% | 16.0% | 19.5% (ja) | 15.2% | 12.2% | -3.0 pp | 12.8% | 13.2% | 1.6% | 10.1% | 16.9% | 11.7% |
| 1 | 0.592675 | 73.5% | 14.6% | 15.5% (zh) | 13.1% | 11.8% | -1.2 pp | 13.4% | 13.8% | 0.2% | 16.1% | 19.1% | 12.8% |
| 2 | 0.754912 | 57.7% | 15.0% | 17.7% (ja) | 14.2% | 12.2% | -2.0 pp | 11.5% | 11.5% | 2.3% | 10.3% | 12.9% | 10.1% |
| 3 | 0.59267 | 72.4% | 14.1% | 16.0% (vi) | 12.3% | 10.6% | -1.7 pp | 10.2% | 12.7% | 0.8% | 15.4% | 16.2% | 12.7% |
| 4 | 0.705785 | 61.8% | 15.0% | 17.4% (ja) | 13.4% | 11.6% | -1.8 pp | 8.7% | 10.5% | 0.8% | 11.7% | 14.7% | 9.5% |
| 5 | 0.651357 | 67.9% | 16.1% | 17.3% (vi) | 14.5% | 11.6% | -2.8 pp | 17.4% | 16.0% | 0.8% | 13.8% | 19.2% | 13.6% |
| 6 | 0.705789 | 60.9% | 16.1% | 18.4% (vi) | 14.7% | 11.9% | -2.8 pp | 12.8% | 15.5% | 2.0% | 12.1% | 16.1% | 12.1% |
| 7 | 0.622465 | 69.4% | 15.3% | 16.7% (ja) | 13.0% | 11.0% | -2.0 pp | 10.7% | 15.9% | 1.0% | 12.9% | 14.7% | 10.1% |
| 8 | 0.651327 | 69.8% | 14.0% | 15.1% (vi) | 12.5% | 11.3% | -1.2 pp | 12.8% | 12.6% | 0.4% | 13.2% | 15.3% | 12.4% |
| 9 | 0.651339 | 67.1% | 15.4% | 17.2% (ja) | 13.8% | 11.9% | -1.9 pp | 10.0% | 11.0% | 0.0% | 12.7% | 17.5% | 12.3% |

Mean flip across seeds: 14.0%-16.1% (mean 15.1%, sd 0.7%). Pivot minus quantile: -3.0 to -1.2 pp (pivot flips fewer decisions in 10/10 splits).

Unsafe promotion range across seeds (share of all held-out incorrect candidates):

| lang | min | max |
|---|---|---|
| de | 8.7% | 17.4% |
| es | 10.5% | 16.0% |
| fr | 0.0% | 2.3% |
| ja | 10.1% | 16.1% |
| vi | 12.9% | 19.2% |
| zh | 9.5% | 13.6% |

### Qwen2.5-7B

| seed | threshold | en coverage | mean flip (6 langs) | worst-lang flip | quantile flip | pivot flip | pivot - quantile | unsafe de | unsafe es | unsafe fr | unsafe ja | unsafe vi | unsafe zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.999997 | 61.0% | 21.6% | 27.2% (vi) | 20.5% | 17.6% | -2.9 pp | 4.9% | 4.1% | 5.6% | 8.8% | 8.0% | 10.5% |
| 1 | 0.99478 | 81.0% | 15.1% | 18.2% (vi) | 14.0% | 16.3% | +2.3 pp | 7.6% | 5.4% | 7.0% | 9.7% | 8.9% | 8.8% |
| 2 | 0.999997 | 62.0% | 22.3% | 28.5% (vi) | 21.0% | 18.5% | -2.5 pp | 4.9% | 5.6% | 4.2% | 8.0% | 7.3% | 9.8% |
| 3 | 0.999877 | 75.4% | 18.5% | 24.6% (vi) | 16.9% | 16.3% | -0.6 pp | 5.1% | 6.1% | 6.1% | 9.6% | 8.0% | 8.6% |
| 4 | 0.999987 | 68.1% | 20.5% | 25.3% (vi) | 19.7% | 18.4% | -1.3 pp | 8.1% | 8.1% | 8.5% | 9.9% | 10.5% | 12.3% |
| 5 | 0.999904 | 72.1% | 18.3% | 22.5% (vi) | 17.3% | 16.9% | -0.4 pp | 4.4% | 4.8% | 5.6% | 9.0% | 5.6% | 9.0% |
| 6 | 0.999983 | 72.5% | 19.2% | 26.4% (vi) | 17.5% | 17.4% | -0.1 pp | 4.0% | 6.0% | 5.5% | 9.3% | 5.3% | 6.8% |
| 7 | 0.999665 | 76.6% | 18.3% | 21.1% (vi) | 16.8% | 17.1% | +0.3 pp | 7.0% | 5.4% | 5.8% | 7.8% | 7.0% | 8.2% |
| 8 | 0.999942 | 71.2% | 18.7% | 24.2% (vi) | 17.2% | 15.9% | -1.3 pp | 4.1% | 7.8% | 6.4% | 7.4% | 5.0% | 7.4% |
| 9 | 0.952574 | 83.5% | 13.1% | 17.5% (vi) | 11.6% | 13.5% | +1.8 pp | 5.6% | 5.6% | 6.4% | 6.2% | 8.5% | 7.7% |

Mean flip across seeds: 13.1%-22.3% (mean 18.6%, sd 2.6%). Pivot minus quantile: -2.9 to +2.3 pp (pivot flips fewer decisions in 7/10 splits).

Unsafe promotion range across seeds (share of all held-out incorrect candidates):

| lang | min | max |
|---|---|---|
| de | 4.0% | 8.1% |
| es | 4.1% | 8.1% |
| fr | 4.2% | 8.5% |
| ja | 6.2% | 9.9% |
| vi | 5.0% | 10.5% |
| zh | 6.8% | 12.3% |

