# Question perplexity vs. confidence shift (big-llama8b)

Verifier: `meta-llama/Llama-3.1-8B-Instruct`. 150 questions, 750 candidates.
Perplexity is mean NLL per token of the question text under the verifier.

| lang | mean logPPL | Δ vs en | mean Δconf |
|---|---|---|---|
| en | 4.415 | +0.000 | +0.0000 |
| de | 3.460 | -0.955 | -0.0104 |
| es | 3.583 | -0.832 | +0.0252 |
| fr | 3.568 | -0.847 | -0.0905 |
| ja | 3.745 | -0.670 | +0.0057 |
| vi | 4.155 | -0.260 | -0.0017 |
| zh | 4.743 | +0.328 | +0.0012 |

- Correlation across the 6 language means: **r = +0.249** (6 points; descriptive only)
- Correlation across 4500 individual (candidate, language) pairs: **r = +0.043**

A strong negative r would mean: the less familiar the question text, the lower the verifier's confidence in identical SQL.
