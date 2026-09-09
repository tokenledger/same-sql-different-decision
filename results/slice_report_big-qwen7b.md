# Cross-lingual verifier slice report (big-qwen7b)

6000 candidates, 4894 correct (81.6% execution accuracy). All candidates generated from the en question; only the question language shown to the verifier varies.

## Ranking and calibration by question language

| lang | AUROC | Brier | ECE | mean conf |
|---|---|---|---|---|
| en | 0.677 | 0.192 | 0.190 | 0.877 |
| de | 0.650 | 0.218 | 0.216 | 0.833 |
| zh | 0.656 | 0.228 | 0.226 | 0.817 |
| es | 0.673 | 0.208 | 0.207 | 0.848 |
| fr | 0.670 | 0.209 | 0.207 | 0.854 |
| ja | 0.655 | 0.225 | 0.224 | 0.830 |
| vi | 0.654 | 0.236 | 0.235 | 0.796 |

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
| 10% | en | 1.000 | 0.099 | 0.360 | 0.360 |
| 10% | de | 1.000 | 0.094 | 0.141 | 0.153 |
| 10% | zh | 1.000 | 0.116 | 0.316 | 0.189 |
| 10% | es | 1.000 | 0.099 | 0.216 | 0.345 |
| 10% | fr | 1.000 | 0.080 | 0.181 | 0.276 |
| 10% | ja | 1.000 | 0.106 | 0.260 | 0.234 |
| 10% | vi | 1.000 | 0.123 | 0.160 | 0.052 |
| 20% | en | 0.000 | 0.184 | 1.000 | 1.000 |
| 20% | de | 0.000 | 0.184 | 1.000 | 1.000 |
| 20% | zh | 0.000 | 0.185 | 0.997 | 1.000 |
| 20% | es | 0.000 | 0.184 | 1.000 | 1.000 |
| 20% | fr | 0.000 | 0.184 | 1.000 | 1.000 |
| 20% | ja | 0.000 | 0.184 | 0.999 | 1.000 |
| 20% | vi | 0.000 | 0.183 | 0.998 | 1.000 |
| 30% | en | 0.000 | 0.184 | 1.000 | 1.000 |
| 30% | de | 0.000 | 0.184 | 1.000 | 1.000 |
| 30% | zh | 0.000 | 0.185 | 0.997 | 1.000 |
| 30% | es | 0.000 | 0.184 | 1.000 | 1.000 |
| 30% | fr | 0.000 | 0.184 | 1.000 | 1.000 |
| 30% | ja | 0.000 | 0.184 | 0.999 | 1.000 |
| 30% | vi | 0.000 | 0.183 | 0.998 | 1.000 |
