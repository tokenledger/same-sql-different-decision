# Cross-lingual score transport: can we recover the ENGLISH decisions?

All transports fitted on a database-disjoint calibration split, none using
correctness labels in the target language. `flips` is the share of identical
SQL whose execute/defer decision differs from the English decision -- the
quantity a transport has to reduce to be worth anything. Coverage parity
alone does not imply decision agreement.

## Llama-3.1-8B

Threshold 0.7549. Held-out: 3005 candidates (2519 correct / 486 incorrect). English coverage 0.544, risk 0.078.

| method | lang | flips | unsafe promoted | lost automation | overlap | coverage | risk |
|---|---|---|---|---|---|---|---|
| raw | de | 13.9% | 12.8% | 5.8% | 0.780 | 0.579 | 0.102 |
| raw | es | 14.9% | 13.2% | 2.6% | 0.778 | 0.646 | 0.096 |
| raw | fr | 14.6% | 1.6% | 14.0% | 0.737 | 0.423 | 0.068 |
| raw | ja | 19.5% | 10.1% | 9.0% | 0.702 | 0.566 | 0.085 |
| raw | vi | 17.3% | 16.9% | 6.5% | 0.736 | 0.595 | 0.107 |
| raw | zh | 15.9% | 11.7% | 7.1% | 0.749 | 0.565 | 0.092 |
| **raw** | **mean** | **16.0%** | **11.0%** | **7.5%** | **0.747** | **0.562** | **0.092** |
| offset | de | 14.7% | 13.4% | 5.7% | 0.770 | 0.587 | 0.103 |
| offset | es | 13.8% | 8.2% | 5.6% | 0.781 | 0.579 | 0.088 |
| offset | fr | 12.1% | 9.5% | 4.7% | 0.805 | 0.579 | 0.094 |
| offset | ja | 20.0% | 12.1% | 8.6% | 0.698 | 0.580 | 0.090 |
| offset | vi | 17.6% | 16.0% | 7.4% | 0.729 | 0.583 | 0.106 |
| offset | zh | 15.7% | 11.1% | 7.3% | 0.750 | 0.556 | 0.089 |
| **offset** | **mean** | **15.7%** | **11.7%** | **6.6%** | **0.756** | **0.577** | **0.095** |
| affine | de | 13.9% | 11.1% | 6.9% | 0.776 | 0.556 | 0.099 |
| affine | es | 13.8% | 8.2% | 5.6% | 0.781 | 0.579 | 0.088 |
| affine | fr | 12.1% | 9.5% | 4.7% | 0.805 | 0.579 | 0.094 |
| affine | ja | 19.9% | 9.7% | 10.4% | 0.690 | 0.542 | 0.083 |
| affine | vi | 17.6% | 16.0% | 7.4% | 0.729 | 0.583 | 0.106 |
| affine | zh | 15.7% | 11.1% | 7.3% | 0.750 | 0.556 | 0.089 |
| **affine** | **mean** | **15.5%** | **10.9%** | **7.1%** | **0.755** | **0.566** | **0.093** |
| quantile | de | 13.8% | 8.4% | 8.4% | 0.771 | 0.524 | 0.091 |
| quantile | es | 13.4% | 7.0% | 6.2% | 0.783 | 0.560 | 0.084 |
| quantile | fr | 11.3% | 7.2% | 6.5% | 0.811 | 0.537 | 0.090 |
| quantile | ja | 19.3% | 8.6% | 10.7% | 0.695 | 0.529 | 0.080 |
| quantile | vi | 17.2% | 15.6% | 7.7% | 0.733 | 0.573 | 0.106 |
| quantile | zh | 16.0% | 8.8% | 9.1% | 0.740 | 0.526 | 0.085 |
| **quantile** | **mean** | **15.2%** | **9.3%** | **8.1%** | **0.755** | **0.542** | **0.089** |
| isotonic | de | 13.6% | 7.0% | 9.1% | 0.770 | 0.505 | 0.085 |
| isotonic | es | 13.2% | 3.9% | 9.4% | 0.773 | 0.492 | 0.072 |
| isotonic | fr | 11.4% | 8.2% | 6.4% | 0.809 | 0.539 | 0.093 |
| isotonic | ja | 19.1% | 8.2% | 11.3% | 0.695 | 0.517 | 0.080 |
| isotonic | vi | 16.8% | 9.1% | 11.8% | 0.719 | 0.487 | 0.089 |
| isotonic | zh | 15.6% | 8.8% | 8.6% | 0.747 | 0.531 | 0.084 |
| **isotonic** | **mean** | **15.0%** | **7.5%** | **9.4%** | **0.752** | **0.512** | **0.084** |

## Qwen2.5-7B

Threshold 1.0000. Held-out: 3005 candidates (2519 correct / 486 incorrect). English coverage 0.610, risk 0.085.

| method | lang | flips | unsafe promoted | lost automation | overlap | coverage | risk |
|---|---|---|---|---|---|---|---|
| raw | de | 21.7% | 4.9% | 17.3% | 0.673 | 0.499 | 0.082 |
| raw | es | 17.4% | 4.1% | 12.5% | 0.737 | 0.537 | 0.074 |
| raw | fr | 20.1% | 5.6% | 14.5% | 0.700 | 0.529 | 0.079 |
| raw | ja | 21.9% | 8.8% | 14.7% | 0.683 | 0.554 | 0.094 |
| raw | vi | 27.2% | 8.0% | 22.1% | 0.595 | 0.462 | 0.087 |
| raw | zh | 21.5% | 10.5% | 13.3% | 0.690 | 0.564 | 0.089 |
| **raw** | **mean** | **21.6%** | **7.0%** | **15.7%** | **0.680** | **0.524** | **0.084** |
| offset | de | 20.7% | 9.9% | 10.7% | 0.709 | 0.608 | 0.087 |
| offset | es | 17.7% | 7.0% | 8.1% | 0.749 | 0.625 | 0.079 |
| offset | fr | 18.2% | 7.8% | 9.1% | 0.741 | 0.611 | 0.082 |
| offset | ja | 21.6% | 12.1% | 11.2% | 0.701 | 0.616 | 0.098 |
| offset | vi | 24.4% | 12.1% | 13.3% | 0.663 | 0.593 | 0.089 |
| offset | zh | 21.5% | 15.8% | 7.9% | 0.712 | 0.669 | 0.098 |
| **offset** | **mean** | **20.7%** | **10.8%** | **10.0%** | **0.712** | **0.620** | **0.089** |
| affine | de | 20.6% | 9.7% | 10.8% | 0.709 | 0.601 | 0.085 |
| affine | es | 17.7% | 7.0% | 8.1% | 0.749 | 0.625 | 0.079 |
| affine | fr | 18.2% | 7.8% | 9.1% | 0.741 | 0.611 | 0.082 |
| affine | ja | 21.4% | 11.9% | 11.2% | 0.702 | 0.614 | 0.098 |
| affine | vi | 24.4% | 12.1% | 13.3% | 0.663 | 0.593 | 0.089 |
| affine | zh | 21.9% | 15.4% | 8.5% | 0.706 | 0.660 | 0.097 |
| **affine** | **mean** | **20.7%** | **10.7%** | **10.2%** | **0.712** | **0.617** | **0.088** |
| quantile | de | 20.7% | 9.9% | 10.7% | 0.709 | 0.608 | 0.087 |
| quantile | es | 17.9% | 7.0% | 8.1% | 0.747 | 0.626 | 0.079 |
| quantile | fr | 17.7% | 6.6% | 9.8% | 0.743 | 0.591 | 0.079 |
| quantile | ja | 21.7% | 11.5% | 12.1% | 0.696 | 0.601 | 0.098 |
| quantile | vi | 23.7% | 12.6% | 12.4% | 0.673 | 0.604 | 0.091 |
| quantile | zh | 21.4% | 16.5% | 7.1% | 0.716 | 0.682 | 0.098 |
| **quantile** | **mean** | **20.5%** | **10.7%** | **10.0%** | **0.714** | **0.619** | **0.089** |
| isotonic | de | 55.8% | 0.0% | 60.7% | 0.086 | 0.053 | 0.063 |
| isotonic | es | 40.6% | 1.2% | 43.0% | 0.342 | 0.217 | 0.067 |
| isotonic | fr | 57.2% | 0.0% | 62.0% | 0.063 | 0.039 | 0.000 |
| isotonic | ja | 49.5% | 0.8% | 53.2% | 0.194 | 0.122 | 0.068 |
| isotonic | vi | 52.2% | 0.0% | 56.3% | 0.147 | 0.092 | 0.047 |
| isotonic | zh | 45.8% | 0.4% | 48.9% | 0.251 | 0.156 | 0.043 |
| **isotonic** | **mean** | **50.2%** | **0.4%** | **54.0%** | **0.180** | **0.113** | **0.048** |

## Summary: mean over the six non-English languages

| verifier | method | flips | unsafe | lost | overlap | coverage spread | risk |
|---|---|---|---|---|---|---|---|
| llama8b | raw | 16.0% | 11.0% | 7.5% | 0.747 | 0.223 | 0.092 |
| llama8b | offset | 15.7% | 11.7% | 6.6% | 0.756 | 0.031 | 0.095 |
| llama8b | affine | 15.5% | 10.9% | 7.1% | 0.755 | 0.040 | 0.093 |
| llama8b | quantile | 15.2% | 9.3% | 8.1% | 0.755 | 0.049 | 0.089 |
| llama8b | isotonic | 15.0% | 7.5% | 9.4% | 0.752 | 0.053 | 0.084 |
| qwen7b | raw | 21.6% | 7.0% | 15.7% | 0.680 | 0.101 | 0.084 |
| qwen7b | offset | 20.7% | 10.8% | 10.0% | 0.712 | 0.076 | 0.089 |
| qwen7b | affine | 20.7% | 10.7% | 10.2% | 0.712 | 0.067 | 0.088 |
| qwen7b | quantile | 20.5% | 10.7% | 10.0% | 0.714 | 0.090 | 0.089 |
| qwen7b | isotonic | 50.2% | 0.4% | 54.0% | 0.180 | 0.178 | 0.048 |

A transport earns a place in the paper only if it cuts `flips` and `unsafe`
without materially raising `risk`. Shrinking `coverage spread` alone is not
sufficient: a rank rule equalises how many queries run while still changing
which ones do.

