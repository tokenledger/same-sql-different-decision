# Cross-lingual verifier significance analysis

## big-llama8b

6000 candidates from 1200 questions, 4894 correct (81.6%), 1106 incorrect. CIs are percentile bootstrap over questions (4000 resamples).

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
| en | 0.675 | — | — | — |
| de | 0.657 | +0.018 | [-0.001, +0.036] | no |
| zh | 0.668 | +0.007 | [-0.014, +0.027] | no |
| es | 0.670 | +0.005 | [-0.010, +0.019] | no |
| fr | 0.670 | +0.005 | [-0.009, +0.018] | no |
| ja | 0.649 | +0.026 | [+0.008, +0.046] | **yes** |
| vi | 0.652 | +0.023 | [+0.002, +0.045] | **yes** |

### Coverage under the English 10%-risk threshold (0.8355)

| lang | realized risk | coverage | gap en−lang | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.100 | 0.469 | — | — | — |
| de | 0.111 | 0.495 | -0.026 | [-0.044, -0.009] | **yes** |
| zh | 0.113 | 0.505 | -0.036 | [-0.056, -0.017] | **yes** |
| es | 0.117 | 0.537 | -0.068 | [-0.085, -0.051] | **yes** |
| fr | 0.087 | 0.358 | +0.110 | [+0.094, +0.127] | **yes** |
| ja | 0.118 | 0.474 | -0.005 | [-0.025, +0.014] | no |
| vi | 0.123 | 0.497 | -0.029 | [-0.048, -0.009] | **yes** |

## big-qwen7b

6000 candidates from 1200 questions, 4894 correct (81.6%), 1106 incorrect. CIs are percentile bootstrap over questions (4000 resamples).

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
| en | 0.677 | — | — | — |
| de | 0.650 | +0.027 | [+0.005, +0.048] | **yes** |
| zh | 0.656 | +0.021 | [-0.005, +0.047] | no |
| es | 0.673 | +0.004 | [-0.017, +0.025] | no |
| fr | 0.670 | +0.008 | [-0.012, +0.028] | no |
| ja | 0.655 | +0.022 | [-0.005, +0.047] | no |
| vi | 0.654 | +0.023 | [-0.008, +0.054] | no |

### Coverage under the English 10%-risk threshold (1.0000)

| lang | realized risk | coverage | gap en−lang | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.099 | 0.360 | — | — | — |
| de | 0.094 | 0.141 | +0.220 | [+0.197, +0.243] | **yes** |
| zh | 0.116 | 0.316 | +0.044 | [+0.023, +0.067] | **yes** |
| es | 0.099 | 0.216 | +0.144 | [+0.124, +0.165] | **yes** |
| fr | 0.080 | 0.181 | +0.179 | [+0.157, +0.201] | **yes** |
| ja | 0.106 | 0.260 | +0.100 | [+0.079, +0.123] | **yes** |
| vi | 0.123 | 0.160 | +0.201 | [+0.178, +0.225] | **yes** |
