# Cross-lingual verifier slice report (big-llama8b)

6000 candidates, 4894 correct (81.6% execution accuracy). All candidates generated from the en question; only the question language shown to the verifier varies.

## Ranking and calibration by question language

| lang | AUROC | Brier | ECE | mean conf |
|---|---|---|---|---|
| en | 0.675 | 0.170 | 0.120 | 0.736 |
| de | 0.657 | 0.174 | 0.125 | 0.738 |
| zh | 0.668 | 0.175 | 0.118 | 0.742 |
| es | 0.670 | 0.163 | 0.105 | 0.767 |
| fr | 0.670 | 0.204 | 0.177 | 0.647 |
| ja | 0.649 | 0.172 | 0.111 | 0.746 |
| vi | 0.652 | 0.173 | 0.127 | 0.753 |

## Paired score shift vs. English (same SQL, same label)

| lang | mean Δconf | mean \|Δconf\| | flipped rank order |
|---|---|---|---|
| de | 0.002 | 0.081 | 10.0% |
| zh | 0.005 | 0.095 | 10.9% |
| es | 0.031 | 0.081 | 10.8% |
| fr | -0.089 | 0.105 | 14.4% |
| ja | 0.009 | 0.100 | 11.9% |
| vi | 0.017 | 0.101 | 13.3% |

## Transferring the English abstention threshold

Threshold fitted on English to hit the target risk, then applied unchanged to each language's scores.

| target risk | lang | threshold | realized risk | coverage | own-language coverage |
|---|---|---|---|---|---|
| 10% | en | 0.835 | 0.100 | 0.469 | 0.469 |
| 10% | de | 0.835 | 0.111 | 0.495 | 0.434 |
| 10% | zh | 0.835 | 0.113 | 0.505 | 0.444 |
| 10% | es | 0.835 | 0.117 | 0.537 | 0.409 |
| 10% | fr | 0.835 | 0.087 | 0.358 | 0.432 |
| 10% | ja | 0.835 | 0.118 | 0.474 | 0.348 |
| 10% | vi | 0.835 | 0.123 | 0.497 | 0.393 |
| 20% | en | 0.006 | 0.184 | 1.000 | 1.000 |
| 20% | de | 0.006 | 0.184 | 1.000 | 1.000 |
| 20% | zh | 0.006 | 0.184 | 0.999 | 1.000 |
| 20% | es | 0.006 | 0.184 | 1.000 | 1.000 |
| 20% | fr | 0.006 | 0.183 | 0.999 | 1.000 |
| 20% | ja | 0.006 | 0.184 | 1.000 | 1.000 |
| 20% | vi | 0.006 | 0.184 | 1.000 | 1.000 |
| 30% | en | 0.006 | 0.184 | 1.000 | 1.000 |
| 30% | de | 0.006 | 0.184 | 1.000 | 1.000 |
| 30% | zh | 0.006 | 0.184 | 0.999 | 1.000 |
| 30% | es | 0.006 | 0.184 | 1.000 | 1.000 |
| 30% | fr | 0.006 | 0.183 | 0.999 | 1.000 |
| 30% | ja | 0.006 | 0.184 | 1.000 | 1.000 |
| 30% | vi | 0.006 | 0.184 | 1.000 | 1.000 |
