# Is the language effect a class-independent offset?

`delta` is the confidence change vs. English on the *same* SQL with the
*same* execution-derived label. If `kappa` (the correct-minus-incorrect
difference) is near zero, the language moves the score scale without
changing discrimination -- which is what allows AUROC to stay flat while a
fixed threshold shifts coverage. Bootstrap resamples DATABASES.

## llama8b

6000 candidates over 163 databases (5053 correct, 947 incorrect).

| lang | shift on correct | shift on incorrect | kappa (difference) | 95% CI | class-independent? |
|---|---|---|---|---|---|
| de | -0.0010 | +0.0163 | -0.0173 | [-0.0364, +0.0031] | yes |
| es | +0.0280 | +0.0461 | -0.0180 | [-0.0330, -0.0025] | yes |
| fr | -0.0875 | -0.1002 | +0.0127 | [-0.0026, +0.0284] | yes |
| ja | +0.0031 | +0.0419 | -0.0387 | [-0.0572, -0.0208] | **no** |
| vi | +0.0106 | +0.0499 | -0.0393 | [-0.0609, -0.0169] | **no** |
| zh | +0.0033 | +0.0173 | -0.0140 | [-0.0341, +0.0064] | yes |

## qwen7b

6000 candidates over 163 databases (5053 correct, 947 incorrect).

| lang | shift on correct | shift on incorrect | kappa (difference) | 95% CI | class-independent? |
|---|---|---|---|---|---|
| de | -0.0425 | -0.0470 | +0.0046 | [-0.0393, +0.0487] | yes |
| es | -0.0293 | -0.0280 | -0.0013 | [-0.0443, +0.0429] | yes |
| fr | -0.0251 | -0.0088 | -0.0163 | [-0.0605, +0.0286] | yes |
| ja | -0.0496 | -0.0324 | -0.0172 | [-0.0627, +0.0292] | yes |
| vi | -0.0804 | -0.0833 | +0.0029 | [-0.0538, +0.0573] | yes |
| zh | -0.0584 | -0.0674 | +0.0090 | [-0.0431, +0.0600] | yes |

## Reading this

A small `kappa` supports describing the language effect as an approximately
additive, class-independent score offset **for these graders on this task**.
It is an empirical description, not a causal explanation, and it does not
rule out language-by-item interaction that averages out across items.

