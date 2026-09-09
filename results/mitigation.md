# Mitigation: does showing English alongside the native question help?

## llama8b

6000 candidates, 4894 correct. English 10%-risk threshold fitted on 81 databases = 0.8355, evaluated on the 82 held out.

### Coverage per language, by condition

| condition | en | de | es | fr | ja | vi | zh | spread |
|---|---|---|---|---|---|---|---|---|
| native | 0.450 | 0.467 | 0.519 | 0.332 | 0.440 | 0.476 | 0.475 | **0.187** |
| bilingual | 0.450 | 0.373 | 0.390 | 0.299 | 0.405 | 0.403 | 0.429 | **0.151** |
| pivot | 0.450 | 0.413 | 0.418 | 0.412 | 0.380 | 0.364 | 0.385 | **0.086** |

### Coverage spread, bootstrapped over questions

| condition | spread | 95% CI | vs native |
|---|---|---|---|
| native | 0.187 | [0.160, 0.216] |  |
| bilingual | 0.151 | [0.127, 0.179] | -0.036 [-0.075, +0.004] no change |
| pivot | 0.086 | [0.064, 0.111] | -0.100 [-0.138, -0.063] **better** |

### AUROC by condition (held-out databases)

| condition | de | es | fr | ja | vi | zh |
|---|---|---|---|---|---|---|
| native | 0.680 | 0.694 | 0.684 | 0.665 | 0.658 | 0.688 |
| bilingual | 0.687 | 0.690 | 0.684 | 0.680 | 0.680 | 0.692 |
| pivot | 0.693 | 0.685 | 0.682 | 0.672 | 0.671 | 0.674 |

English reference AUROC: 0.689

## qwen7b

6000 candidates, 4894 correct. English 10%-risk threshold fitted on 81 databases = 1.0000, evaluated on the 82 held out.

### Coverage per language, by condition

| condition | en | de | es | fr | ja | vi | zh | spread |
|---|---|---|---|---|---|---|---|---|
| native | 0.304 | 0.098 | 0.174 | 0.135 | 0.199 | 0.103 | 0.260 | **0.206** |
| bilingual | 0.304 | 0.393 | 0.381 | 0.386 | 0.369 | 0.314 | 0.370 | **0.089** |
| pivot | 0.304 | 0.244 | 0.242 | 0.264 | 0.238 | 0.219 | 0.240 | **0.085** |

### Coverage spread, bootstrapped over questions

| condition | spread | 95% CI | vs native |
|---|---|---|---|
| native | 0.206 | [0.180, 0.241] |  |
| bilingual | 0.089 | [0.072, 0.119] | -0.115 [-0.153, -0.078] **better** |
| pivot | 0.085 | [0.063, 0.114] | -0.123 [-0.163, -0.082] **better** |

### AUROC by condition (held-out databases)

| condition | de | es | fr | ja | vi | zh |
|---|---|---|---|---|---|---|
| native | 0.666 | 0.693 | 0.685 | 0.650 | 0.675 | 0.661 |
| bilingual | 0.691 | 0.700 | 0.706 | 0.682 | 0.699 | 0.682 |
| pivot | 0.686 | 0.697 | 0.680 | 0.666 | 0.693 | 0.664 |

English reference AUROC: 0.697

