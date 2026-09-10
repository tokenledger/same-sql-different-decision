# Cross-lingual verifier slice report (big-llama8b)

6000 candidates, 5053 correct (84.2% execution accuracy). All candidates generated from the en question; only the question language shown to the verifier varies.

## Ranking and calibration by question language

| lang | AUROC | Brier | ECE | mean conf |
|---|---|---|---|---|
| en | 0.711 | 0.154 | 0.116 | 0.736 |
| de | 0.682 | 0.161 | 0.122 | 0.738 |
| zh | 0.703 | 0.158 | 0.111 | 0.742 |
| es | 0.703 | 0.147 | 0.096 | 0.767 |
| fr | 0.698 | 0.195 | 0.195 | 0.647 |
| ja | 0.682 | 0.156 | 0.107 | 0.746 |
| vi | 0.678 | 0.158 | 0.124 | 0.753 |

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
| 10% | en | 0.706 | 0.100 | 0.609 | 0.609 |
| 10% | de | 0.706 | 0.109 | 0.650 | 0.555 |
| 10% | zh | 0.706 | 0.103 | 0.639 | 0.602 |
| 10% | es | 0.706 | 0.109 | 0.696 | 0.647 |
| 10% | fr | 0.706 | 0.080 | 0.492 | 0.640 |
| 10% | ja | 0.706 | 0.110 | 0.653 | 0.619 |
| 10% | vi | 0.706 | 0.118 | 0.664 | 0.515 |
| 20% | en | 0.006 | 0.158 | 1.000 | 1.000 |
| 20% | de | 0.006 | 0.158 | 1.000 | 1.000 |
| 20% | zh | 0.006 | 0.158 | 0.999 | 1.000 |
| 20% | es | 0.006 | 0.158 | 1.000 | 1.000 |
| 20% | fr | 0.006 | 0.157 | 0.999 | 1.000 |
| 20% | ja | 0.006 | 0.158 | 1.000 | 1.000 |
| 20% | vi | 0.006 | 0.158 | 1.000 | 1.000 |
| 30% | en | 0.006 | 0.158 | 1.000 | 1.000 |
| 30% | de | 0.006 | 0.158 | 1.000 | 1.000 |
| 30% | zh | 0.006 | 0.158 | 0.999 | 1.000 |
| 30% | es | 0.006 | 0.158 | 1.000 | 1.000 |
| 30% | fr | 0.006 | 0.157 | 0.999 | 1.000 |
| 30% | ja | 0.006 | 0.158 | 1.000 | 1.000 |
| 30% | vi | 0.006 | 0.158 | 1.000 | 1.000 |
