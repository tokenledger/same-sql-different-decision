# Cross-lingual verifier slice report (local-llama8b)

60 candidates, 48 correct (80.0% execution accuracy). All candidates generated from the en question; only the question language shown to the verifier varies.

## Ranking and calibration by question language

| lang | AUROC | Brier | ECE | mean conf |
|---|---|---|---|---|
| en | 0.769 | 0.160 | 0.154 | 0.739 |
| de | 0.786 | 0.152 | 0.173 | 0.720 |
| zh | 0.774 | 0.146 | 0.191 | 0.765 |

## Paired score shift vs. English (same SQL, same label)

| lang | mean Δconf | mean \|Δconf\| | flipped rank order |
|---|---|---|---|
| de | -0.019 | 0.058 | 3.3% |
| zh | 0.026 | 0.055 | 3.3% |

## Transferring the English abstention threshold

Threshold fitted on English to hit the target risk, then applied unchanged to each language's scores.

| target risk | lang | threshold | realized risk | coverage | own-language coverage |
|---|---|---|---|---|---|
| 10% | en | 0.818 | 0.062 | 0.533 | 0.533 |
| 10% | de | 0.818 | 0.000 | 0.483 | 0.683 |
| 10% | zh | 0.818 | 0.059 | 0.567 | 0.583 |
| 20% | en | 0.002 | 0.200 | 1.000 | 1.000 |
| 20% | de | 0.002 | 0.200 | 1.000 | 1.000 |
| 20% | zh | 0.002 | 0.200 | 1.000 | 1.000 |
| 30% | en | 0.002 | 0.200 | 1.000 | 1.000 |
| 30% | de | 0.002 | 0.200 | 1.000 | 1.000 |
| 30% | zh | 0.002 | 0.200 | 1.000 | 1.000 |
