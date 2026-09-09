# Cross-lingual verifier significance analysis

## big-llama8b

6000 candidates from 1200 questions, 5053 correct (84.2%), 947 incorrect. CIs are percentile bootstrap over questions (4000 resamples).

### Paired confidence shift vs. English (identical SQL and label)

| lang | mean Δconf | 95% CI | significant |
|---|---|---|---|
| de | +0.0018 | [-0.0053, +0.0092] | no |
| zh | +0.0055 | [-0.0025, +0.0136] | no |
| es | +0.0309 | [+0.0243, +0.0377] | **yes** |
| fr | -0.0895 | [-0.0958, -0.0834] | **yes** |
| ja | +0.0093 | [+0.0011, +0.0175] | **yes** |
| vi | +0.0168 | [+0.0085, +0.0251] | **yes** |

### AUROC gap vs. English

| lang | AUROC | gap en−lang | 95% CI | significant |
|---|---|---|---|---|
| en | 0.711 | — | — | — |
| de | 0.682 | +0.029 | [+0.009, +0.047] | **yes** |
| zh | 0.703 | +0.008 | [-0.013, +0.028] | no |
| es | 0.703 | +0.008 | [-0.008, +0.024] | no |
| fr | 0.698 | +0.013 | [-0.001, +0.026] | no |
| ja | 0.682 | +0.028 | [+0.009, +0.048] | **yes** |
| vi | 0.678 | +0.033 | [+0.009, +0.055] | **yes** |

### Coverage under the English 10%-risk threshold (0.7058)

| lang | realized risk | coverage | gap en−lang | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.100 | 0.609 | — | — | — |
| de | 0.109 | 0.650 | -0.041 | [-0.058, -0.022] | **yes** |
| zh | 0.103 | 0.639 | -0.030 | [-0.049, -0.011] | **yes** |
| es | 0.109 | 0.696 | -0.087 | [-0.106, -0.069] | **yes** |
| fr | 0.080 | 0.492 | +0.116 | [+0.100, +0.135] | **yes** |
| ja | 0.110 | 0.653 | -0.044 | [-0.065, -0.024] | **yes** |
| vi | 0.118 | 0.664 | -0.055 | [-0.076, -0.034] | **yes** |

## big-qwen7b

6000 candidates from 1200 questions, 5053 correct (84.2%), 947 incorrect. CIs are percentile bootstrap over questions (4000 resamples).

### Paired confidence shift vs. English (identical SQL and label)

| lang | mean Δconf | 95% CI | significant |
|---|---|---|---|
| de | -0.0432 | [-0.0577, -0.0284] | **yes** |
| zh | -0.0598 | [-0.0777, -0.0426] | **yes** |
| es | -0.0291 | [-0.0432, -0.0162] | **yes** |
| fr | -0.0225 | [-0.0357, -0.0094] | **yes** |
| ja | -0.0469 | [-0.0641, -0.0301] | **yes** |
| vi | -0.0809 | [-0.0998, -0.0627] | **yes** |

### AUROC gap vs. English

| lang | AUROC | gap en−lang | 95% CI | significant |
|---|---|---|---|---|
| en | 0.725 | — | — | — |
| de | 0.694 | +0.031 | [+0.008, +0.054] | **yes** |
| zh | 0.703 | +0.022 | [-0.004, +0.049] | no |
| es | 0.714 | +0.011 | [-0.011, +0.034] | no |
| fr | 0.704 | +0.021 | [+0.000, +0.042] | **yes** |
| ja | 0.695 | +0.030 | [+0.003, +0.056] | **yes** |
| vi | 0.685 | +0.040 | [+0.008, +0.069] | **yes** |

### Coverage under the English 10%-risk threshold (1.0000)

| lang | realized risk | coverage | gap en−lang | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.099 | 0.698 | — | — | — |
| de | 0.093 | 0.603 | +0.095 | [+0.072, +0.117] | **yes** |
| zh | 0.098 | 0.627 | +0.071 | [+0.048, +0.094] | **yes** |
| es | 0.094 | 0.639 | +0.059 | [+0.038, +0.080] | **yes** |
| fr | 0.093 | 0.658 | +0.040 | [+0.021, +0.059] | **yes** |
| ja | 0.099 | 0.644 | +0.054 | [+0.031, +0.077] | **yes** |
| vi | 0.093 | 0.560 | +0.138 | [+0.113, +0.163] | **yes** |
