# Cross-lingual verifier slice report (colab-llama8b)

400 candidates, 341 correct (85.2% execution accuracy). All candidates generated from the en question; only the question language shown to the verifier varies.

## Ranking and calibration by question language

| lang | AUROC | Brier | ECE | mean conf |
|---|---|---|---|---|
| en | 0.725 | 0.143 | 0.138 | 0.749 |
| de | 0.701 | 0.153 | 0.116 | 0.737 |
| zh | 0.729 | 0.140 | 0.120 | 0.754 |
| es | 0.698 | 0.141 | 0.128 | 0.767 |
| fr | 0.691 | 0.205 | 0.231 | 0.622 |
| ja | 0.705 | 0.143 | 0.113 | 0.743 |
| vi | 0.688 | 0.145 | 0.116 | 0.752 |

## Paired score shift vs. English (same SQL, same label)

| lang | mean Δconf | mean \|Δconf\| | flipped rank order |
|---|---|---|---|
| de | -0.012 | 0.083 | 10.5% |
| zh | 0.005 | 0.082 | 9.5% |
| es | 0.018 | 0.084 | 14.2% |
| fr | -0.127 | 0.139 | 21.0% |
| ja | -0.006 | 0.087 | 10.5% |
| vi | 0.003 | 0.078 | 8.8% |

## Transferring the English abstention threshold

Threshold fitted on English to hit the target risk, then applied unchanged to each language's scores.

| target risk | lang | threshold | realized risk | coverage | own-language coverage |
|---|---|---|---|---|---|
| 10% | en | 0.755 | 0.096 | 0.570 | 0.570 |
| 10% | de | 0.755 | 0.087 | 0.575 | 0.677 |
| 10% | zh | 0.755 | 0.079 | 0.570 | 0.800 |
| 10% | es | 0.755 | 0.108 | 0.623 | 0.557 |
| 10% | fr | 0.755 | 0.074 | 0.407 | 0.657 |
| 10% | ja | 0.755 | 0.074 | 0.575 | 0.748 |
| 10% | vi | 0.755 | 0.086 | 0.580 | 0.660 |
| 20% | en | 0.026 | 0.147 | 1.000 | 1.000 |
| 20% | de | 0.026 | 0.147 | 1.000 | 1.000 |
| 20% | zh | 0.026 | 0.147 | 1.000 | 1.000 |
| 20% | es | 0.026 | 0.145 | 0.998 | 1.000 |
| 20% | fr | 0.026 | 0.145 | 0.998 | 1.000 |
| 20% | ja | 0.026 | 0.147 | 1.000 | 1.000 |
| 20% | vi | 0.026 | 0.147 | 1.000 | 1.000 |
| 30% | en | 0.026 | 0.147 | 1.000 | 1.000 |
| 30% | de | 0.026 | 0.147 | 1.000 | 1.000 |
| 30% | zh | 0.026 | 0.147 | 1.000 | 1.000 |
| 30% | es | 0.026 | 0.145 | 0.998 | 1.000 |
| 30% | fr | 0.026 | 0.145 | 0.998 | 1.000 |
| 30% | ja | 0.026 | 0.147 | 1.000 | 1.000 |
| 30% | vi | 0.026 | 0.147 | 1.000 | 1.000 |
