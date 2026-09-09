# Mitigation: does showing English alongside the native question help?

## llama8b

6000 candidates, 5053 correct. English 10%-risk threshold fitted on 81 databases = 0.7549, evaluated on the 82 held out.

### Coverage per language, by condition

| condition | en | de | es | fr | ja | vi | zh | spread |
|---|---|---|---|---|---|---|---|---|
| native | 0.544 | 0.579 | 0.646 | 0.423 | 0.566 | 0.595 | 0.565 | **0.223** |
| bilingual | 0.544 | 0.486 | 0.527 | 0.409 | 0.519 | 0.527 | 0.526 | **0.135** |
| pivot | 0.544 | 0.518 | 0.518 | 0.510 | 0.485 | 0.446 | 0.494 | **0.099** |

### Coverage spread, bootstrapped over questions

| condition | spread | 95% CI | vs native |
|---|---|---|---|
| native | 0.223 | [0.194, 0.253] |  |
| bilingual | 0.135 | [0.113, 0.160] | -0.088 [-0.126, -0.050] **better** |
| pivot | 0.099 | [0.070, 0.127] | -0.125 [-0.169, -0.084] **better** |

### AUROC by condition (held-out databases)

| condition | de | es | fr | ja | vi | zh |
|---|---|---|---|---|---|---|
| native | 0.710 | 0.732 | 0.714 | 0.704 | 0.687 | 0.726 |
| bilingual | 0.728 | 0.734 | 0.724 | 0.723 | 0.717 | 0.733 |
| pivot | 0.732 | 0.722 | 0.719 | 0.706 | 0.698 | 0.717 |

English reference AUROC: 0.735

## qwen7b

6000 candidates, 5053 correct. English 10%-risk threshold fitted on 81 databases = 1.0000, evaluated on the 82 held out.

### Coverage per language, by condition

| condition | en | de | es | fr | ja | vi | zh | spread |
|---|---|---|---|---|---|---|---|---|
| native | 0.610 | 0.499 | 0.537 | 0.529 | 0.554 | 0.462 | 0.564 | **0.148** |
| bilingual | 0.610 | 0.656 | 0.656 | 0.663 | 0.670 | 0.617 | 0.681 | **0.071** |
| pivot | 0.610 | 0.533 | 0.553 | 0.537 | 0.481 | 0.459 | 0.517 | **0.151** |

### Coverage spread, bootstrapped over questions

| condition | spread | 95% CI | vs native |
|---|---|---|---|
| native | 0.148 | [0.112, 0.183] |  |
| bilingual | 0.071 | [0.055, 0.101] | -0.071 [-0.111, -0.030] **better** |
| pivot | 0.151 | [0.121, 0.185] | +0.005 [-0.041, +0.055] no change |

### AUROC by condition (held-out databases)

| condition | de | es | fr | ja | vi | zh |
|---|---|---|---|---|---|---|
| native | 0.716 | 0.743 | 0.729 | 0.701 | 0.706 | 0.717 |
| bilingual | 0.744 | 0.760 | 0.763 | 0.739 | 0.752 | 0.739 |
| pivot | 0.740 | 0.745 | 0.737 | 0.727 | 0.740 | 0.723 |

English reference AUROC: 0.754

