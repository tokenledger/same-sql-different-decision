# Cross-lingual verifier slice report (big-qwen7b)

6000 candidates, 5053 correct (84.2% execution accuracy). All candidates generated from the en question; only the question language shown to the verifier varies.

## Ranking and calibration by question language

| lang | AUROC | Brier | ECE | mean conf |
|---|---|---|---|---|
| en | 0.725 | 0.167 | 0.165 | 0.877 |
| de | 0.694 | 0.196 | 0.194 | 0.833 |
| zh | 0.703 | 0.205 | 0.203 | 0.817 |
| es | 0.714 | 0.186 | 0.184 | 0.848 |
| fr | 0.704 | 0.187 | 0.186 | 0.854 |
| ja | 0.695 | 0.202 | 0.202 | 0.830 |
| vi | 0.685 | 0.218 | 0.219 | 0.796 |

## Paired score shift vs. English (same SQL, same label)

| lang | mean Δconf | mean \|Δconf\| | flipped rank order |
|---|---|---|---|
| de | -0.043 | 0.108 | 10.8% |
| zh | -0.060 | 0.140 | 14.0% |
| es | -0.029 | 0.096 | 9.6% |
| fr | -0.023 | 0.092 | 9.1% |
| ja | -0.047 | 0.133 | 13.6% |
| vi | -0.081 | 0.157 | 15.8% |

## Transferring the English abstention threshold

Threshold fitted on English to hit the target risk, then applied unchanged to each language's scores.

| target risk | lang | threshold | realized risk | coverage | own-language coverage |
|---|---|---|---|---|---|
| 10% | en | 1.000 | 0.099 | 0.698 | 0.698 |
| 10% | de | 1.000 | 0.093 | 0.603 | 0.683 |
| 10% | zh | 1.000 | 0.098 | 0.627 | 0.665 |
| 10% | es | 1.000 | 0.094 | 0.639 | 0.706 |
| 10% | fr | 1.000 | 0.093 | 0.658 | 0.709 |
| 10% | ja | 1.000 | 0.099 | 0.644 | 0.645 |
| 10% | vi | 1.000 | 0.093 | 0.560 | 0.642 |
| 20% | en | 0.000 | 0.158 | 1.000 | 1.000 |
| 20% | de | 0.000 | 0.158 | 1.000 | 1.000 |
| 20% | zh | 0.000 | 0.158 | 0.997 | 1.000 |
| 20% | es | 0.000 | 0.158 | 1.000 | 1.000 |
| 20% | fr | 0.000 | 0.158 | 1.000 | 1.000 |
| 20% | ja | 0.000 | 0.157 | 0.999 | 1.000 |
| 20% | vi | 0.000 | 0.157 | 0.998 | 1.000 |
| 30% | en | 0.000 | 0.158 | 1.000 | 1.000 |
| 30% | de | 0.000 | 0.158 | 1.000 | 1.000 |
| 30% | zh | 0.000 | 0.158 | 0.997 | 1.000 |
| 30% | es | 0.000 | 0.158 | 1.000 | 1.000 |
| 30% | fr | 0.000 | 0.158 | 1.000 | 1.000 |
| 30% | ja | 0.000 | 0.157 | 0.999 | 1.000 |
| 30% | vi | 0.000 | 0.157 | 0.998 | 1.000 |
