# Cross-lingual verifier slice report (local)

60 candidates, 48 correct (80.0% execution accuracy). All candidates generated from the en question; only the question language shown to the verifier varies.

## Ranking and calibration by question language

| lang | AUROC | Brier | ECE | mean conf |
|---|---|---|---|---|
| en | 0.806 | 0.164 | 0.165 | 0.965 |
| de | 0.679 | 0.175 | 0.193 | 0.937 |
| zh | 0.750 | 0.166 | 0.168 | 0.968 |

## Paired score shift vs. English (same SQL, same label)

| lang | mean Δconf | mean \|Δconf\| | flipped rank order |
|---|---|---|---|
| de | -0.028 | 0.033 | 0.0% |
| zh | 0.003 | 0.004 | 0.0% |

## Transferring the English abstention threshold

Threshold fitted on English to hit the target risk, then applied unchanged to each language's scores.

| target risk | lang | threshold | realized risk | coverage | own-language coverage |
|---|---|---|---|---|---|
| 10% | en | 0.999 | 0.093 | 0.717 | 0.717 |
| 10% | de | 0.999 | 0.094 | 0.533 | 0.533 |
| 10% | zh | 0.999 | 0.104 | 0.800 | 0.783 |
| 20% | en | 0.029 | 0.200 | 1.000 | 1.000 |
| 20% | de | 0.029 | 0.172 | 0.967 | 1.000 |
| 20% | zh | 0.029 | 0.200 | 1.000 | 1.000 |
| 30% | en | 0.029 | 0.200 | 1.000 | 1.000 |
| 30% | de | 0.029 | 0.172 | 0.967 | 1.000 |
| 30% | zh | 0.029 | 0.200 | 1.000 | 1.000 |
