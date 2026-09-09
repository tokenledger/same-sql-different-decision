# Cross-lingual verifier slice report (colab-qwen1.5b)

400 candidates, 341 correct (85.2% execution accuracy). All candidates generated from the en question; only the question language shown to the verifier varies.

## Ranking and calibration by question language

| lang | AUROC | Brier | ECE | mean conf |
|---|---|---|---|---|
| en | 0.573 | 0.147 | 0.144 | 0.997 |
| de | 0.526 | 0.170 | 0.183 | 0.957 |
| zh | 0.555 | 0.145 | 0.148 | 0.992 |
| es | 0.586 | 0.145 | 0.143 | 0.995 |
| fr | 0.552 | 0.157 | 0.162 | 0.975 |
| ja | 0.502 | 0.145 | 0.146 | 0.987 |
| vi | 0.508 | 0.148 | 0.147 | 0.985 |

## Paired score shift vs. English (same SQL, same label)

| lang | mean Δconf | mean \|Δconf\| | flipped rank order |
|---|---|---|---|
| de | -0.040 | 0.043 | 3.2% |
| zh | -0.004 | 0.008 | 0.0% |
| es | -0.002 | 0.005 | 0.0% |
| fr | -0.021 | 0.024 | 2.0% |
| ja | -0.010 | 0.014 | 1.0% |
| vi | -0.011 | 0.014 | 1.0% |

## Transferring the English abstention threshold

Threshold fitted on English to hit the target risk, then applied unchanged to each language's scores.

| target risk | lang | threshold | realized risk | coverage | own-language coverage |
|---|---|---|---|---|---|
| 10% | en | 1.000 | 0.100 | 0.175 | 0.175 |
| 10% | de | 1.000 | 0.000 | 0.025 | 0.312 |
| 10% | zh | 1.000 | 0.031 | 0.163 | 0.242 |
| 10% | es | 1.000 | 0.000 | 0.055 | 0.460 |
| 10% | fr | 1.000 | 0.000 | 0.000 | 0.398 |
| 10% | ja | 1.000 | 0.000 | 0.043 | 0.087 |
| 10% | vi | 1.000 | 0.091 | 0.055 | 0.230 |
| 20% | en | 0.818 | 0.147 | 1.000 | 1.000 |
| 20% | de | 0.818 | 0.156 | 0.927 | 1.000 |
| 20% | zh | 0.818 | 0.143 | 0.995 | 1.000 |
| 20% | es | 0.818 | 0.148 | 0.998 | 1.000 |
| 20% | fr | 0.818 | 0.153 | 0.965 | 1.000 |
| 20% | ja | 0.818 | 0.148 | 0.983 | 1.000 |
| 20% | vi | 0.818 | 0.147 | 0.985 | 1.000 |
| 30% | en | 0.818 | 0.147 | 1.000 | 1.000 |
| 30% | de | 0.818 | 0.156 | 0.927 | 1.000 |
| 30% | zh | 0.818 | 0.143 | 0.995 | 1.000 |
| 30% | es | 0.818 | 0.148 | 0.998 | 1.000 |
| 30% | fr | 0.818 | 0.153 | 0.965 | 1.000 |
| 30% | ja | 0.818 | 0.148 | 0.983 | 1.000 |
| 30% | vi | 0.818 | 0.147 | 0.985 | 1.000 |
