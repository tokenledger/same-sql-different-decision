# Cross-lingual score transport: can we recover the ENGLISH decisions?

All transports fitted on a database-disjoint calibration split, none using
correctness labels in the target language. `flips` is the share of identical
SQL whose execute/defer decision differs from the English decision -- the
quantity a transport has to reduce to be worth anything. Coverage parity
alone does not imply decision agreement.

## Llama-3.1-8B

Threshold 0.8355. Held-out: 3005 candidates (2436 correct / 569 incorrect). English coverage 0.450, risk 0.094.

| method | lang | flips | unsafe promoted | lost automation | overlap | coverage | risk |
|---|---|---|---|---|---|---|---|
| raw | de | 12.1% | 7.4% | 5.5% | 0.767 | 0.467 | 0.103 |
| raw | es | 13.7% | 9.5% | 3.7% | 0.752 | 0.519 | 0.108 |
| raw | fr | 13.7% | 0.0% | 13.6% | 0.702 | 0.332 | 0.076 |
| raw | ja | 16.5% | 7.6% | 9.7% | 0.687 | 0.440 | 0.107 |
| raw | vi | 14.4% | 12.0% | 6.0% | 0.731 | 0.476 | 0.115 |
| raw | zh | 14.8% | 9.1% | 6.4% | 0.725 | 0.475 | 0.106 |
| **raw** | **mean** | **14.2%** | **7.6%** | **7.5%** | **0.727** | **0.451** | **0.103** |
| offset | de | 12.4% | 7.7% | 5.5% | 0.762 | 0.470 | 0.104 |
| offset | es | 10.8% | 5.3% | 5.5% | 0.786 | 0.456 | 0.101 |
| offset | fr | 9.3% | 5.3% | 4.3% | 0.816 | 0.465 | 0.103 |
| offset | ja | 16.8% | 7.9% | 9.7% | 0.683 | 0.443 | 0.108 |
| offset | vi | 14.4% | 10.9% | 6.8% | 0.727 | 0.461 | 0.112 |
| offset | zh | 14.6% | 9.1% | 6.9% | 0.724 | 0.462 | 0.104 |
| **offset** | **mean** | **13.1%** | **7.7%** | **6.4%** | **0.750** | **0.460** | **0.105** |
| affine | de | 12.4% | 7.7% | 5.5% | 0.762 | 0.470 | 0.104 |
| affine | es | 10.8% | 5.3% | 5.5% | 0.786 | 0.456 | 0.101 |
| affine | fr | 9.3% | 5.3% | 4.3% | 0.816 | 0.465 | 0.103 |
| affine | ja | 16.8% | 7.9% | 9.7% | 0.683 | 0.443 | 0.108 |
| affine | vi | 14.4% | 10.9% | 6.8% | 0.727 | 0.461 | 0.112 |
| affine | zh | 14.6% | 9.1% | 6.9% | 0.724 | 0.462 | 0.104 |
| **affine** | **mean** | **13.1%** | **7.7%** | **6.4%** | **0.750** | **0.460** | **0.105** |
| quantile | de | 11.6% | 4.7% | 7.1% | 0.766 | 0.426 | 0.091 |
| quantile | es | 10.9% | 4.7% | 6.7% | 0.780 | 0.436 | 0.099 |
| quantile | fr | 9.2% | 4.6% | 5.6% | 0.813 | 0.441 | 0.104 |
| quantile | ja | 16.7% | 5.6% | 11.2% | 0.676 | 0.415 | 0.102 |
| quantile | vi | 13.9% | 8.4% | 7.1% | 0.733 | 0.452 | 0.104 |
| quantile | zh | 14.1% | 7.0% | 7.6% | 0.727 | 0.441 | 0.094 |
| **quantile** | **mean** | **12.8%** | **5.9%** | **7.5%** | **0.749** | **0.435** | **0.099** |
| isotonic | de | 12.0% | 3.9% | 8.9% | 0.752 | 0.397 | 0.089 |
| isotonic | es | 12.8% | 1.4% | 11.2% | 0.729 | 0.366 | 0.082 |
| isotonic | fr | 9.2% | 4.2% | 6.2% | 0.811 | 0.431 | 0.105 |
| isotonic | ja | 16.7% | 5.8% | 10.6% | 0.680 | 0.426 | 0.102 |
| isotonic | vi | 15.0% | 3.5% | 11.9% | 0.695 | 0.382 | 0.097 |
| isotonic | zh | 14.7% | 1.8% | 12.1% | 0.692 | 0.360 | 0.069 |
| **isotonic** | **mean** | **13.4%** | **3.4%** | **10.2%** | **0.726** | **0.394** | **0.091** |

## Qwen2.5-7B

Threshold 1.0000. Held-out: 3005 candidates (2436 correct / 569 incorrect). English coverage 0.304, risk 0.089.

| method | lang | flips | unsafe promoted | lost automation | overlap | coverage | risk |
|---|---|---|---|---|---|---|---|
| raw | de | 22.3% | 0.9% | 23.9% | 0.286 | 0.098 | 0.082 |
| raw | es | 18.2% | 3.0% | 17.4% | 0.449 | 0.174 | 0.100 |
| raw | fr | 19.1% | 1.8% | 19.7% | 0.393 | 0.135 | 0.074 |
| raw | ja | 18.4% | 4.7% | 16.0% | 0.464 | 0.199 | 0.104 |
| raw | vi | 22.7% | 0.4% | 24.3% | 0.284 | 0.103 | 0.106 |
| raw | zh | 17.5% | 6.2% | 12.4% | 0.526 | 0.260 | 0.114 |
| **raw** | **mean** | **19.7%** | **2.8%** | **19.0%** | **0.400** | **0.161** | **0.097** |
| offset | de | 18.7% | 8.8% | 8.8% | 0.550 | 0.341 | 0.117 |
| offset | es | 15.8% | 8.3% | 5.4% | 0.617 | 0.362 | 0.101 |
| offset | fr | 14.4% | 6.3% | 8.3% | 0.613 | 0.294 | 0.101 |
| offset | ja | 21.0% | 12.7% | 8.5% | 0.524 | 0.370 | 0.129 |
| offset | vi | 23.8% | 14.2% | 8.0% | 0.490 | 0.390 | 0.109 |
| offset | zh | 23.6% | 17.0% | 4.6% | 0.531 | 0.464 | 0.126 |
| **offset** | **mean** | **19.6%** | **11.2%** | **7.3%** | **0.554** | **0.370** | **0.114** |
| affine | de | 17.9% | 7.9% | 9.6% | 0.553 | 0.317 | 0.117 |
| affine | es | 15.8% | 7.4% | 6.6% | 0.606 | 0.340 | 0.098 |
| affine | fr | 14.4% | 5.3% | 9.4% | 0.602 | 0.276 | 0.098 |
| affine | ja | 18.1% | 9.7% | 9.6% | 0.550 | 0.319 | 0.126 |
| affine | vi | 21.4% | 11.1% | 9.7% | 0.498 | 0.335 | 0.104 |
| affine | zh | 17.9% | 12.5% | 6.3% | 0.584 | 0.376 | 0.128 |
| **affine** | **mean** | **17.6%** | **9.0%** | **8.5%** | **0.565** | **0.327** | **0.112** |
| quantile | de | 17.7% | 7.2% | 10.3% | 0.548 | 0.301 | 0.115 |
| quantile | es | 15.7% | 5.6% | 8.0% | 0.596 | 0.315 | 0.093 |
| quantile | fr | 14.2% | 6.3% | 7.7% | 0.623 | 0.305 | 0.101 |
| quantile | ja | 17.5% | 9.1% | 10.6% | 0.547 | 0.294 | 0.129 |
| quantile | vi | 21.0% | 8.6% | 11.8% | 0.482 | 0.297 | 0.102 |
| quantile | zh | 17.7% | 8.1% | 10.4% | 0.546 | 0.297 | 0.116 |
| **quantile** | **mean** | **17.3%** | **7.5%** | **9.8%** | **0.557** | **0.302** | **0.109** |
| isotonic | de | 25.9% | 0.7% | 28.4% | 0.160 | 0.053 | 0.063 |
| isotonic | es | 25.9% | 0.9% | 28.4% | 0.155 | 0.050 | 0.053 |
| isotonic | fr | 26.8% | 0.0% | 29.7% | 0.120 | 0.037 | 0.000 |
| isotonic | ja | 27.7% | 0.2% | 30.8% | 0.093 | 0.029 | 0.034 |
| isotonic | vi | 28.1% | 0.0% | 31.4% | 0.081 | 0.026 | 0.077 |
| isotonic | zh | 29.1% | 0.0% | 32.6% | 0.045 | 0.014 | 0.024 |
| **isotonic** | **mean** | **27.2%** | **0.3%** | **30.2%** | **0.109** | **0.035** | **0.042** |

## Summary: mean over the six non-English languages

| verifier | method | flips | unsafe | lost | overlap | coverage spread | risk |
|---|---|---|---|---|---|---|---|
| llama8b | raw | 14.2% | 7.6% | 7.5% | 0.727 | 0.187 | 0.103 |
| llama8b | offset | 13.1% | 7.7% | 6.4% | 0.750 | 0.027 | 0.105 |
| llama8b | affine | 13.1% | 7.7% | 6.4% | 0.750 | 0.027 | 0.105 |
| llama8b | quantile | 12.8% | 5.9% | 7.5% | 0.749 | 0.038 | 0.099 |
| llama8b | isotonic | 13.4% | 3.4% | 10.2% | 0.726 | 0.071 | 0.091 |
| qwen7b | raw | 19.7% | 2.8% | 19.0% | 0.400 | 0.162 | 0.097 |
| qwen7b | offset | 19.6% | 11.2% | 7.3% | 0.554 | 0.170 | 0.114 |
| qwen7b | affine | 17.6% | 9.0% | 8.5% | 0.565 | 0.100 | 0.112 |
| qwen7b | quantile | 17.3% | 7.5% | 9.8% | 0.557 | 0.021 | 0.109 |
| qwen7b | isotonic | 27.2% | 0.3% | 30.2% | 0.109 | 0.039 | 0.042 |

A transport earns a place in the paper only if it cuts `flips` and `unsafe`
without materially raising `risk`. Shrinking `coverage spread` alone is not
sufficient: a rank rule equalises how many queries run while still changing
which ones do.

