# Is the language effect a class-independent offset?

`delta` is the confidence change vs. English on the *same* SQL with the
*same* execution-derived label. If `kappa` (the correct-minus-incorrect
difference) is near zero, the language moves the score scale without
changing discrimination -- which is what allows AUROC to stay flat while a
fixed threshold shifts coverage. Bootstrap resamples DATABASES.

## llama8b

6000 candidates over 163 databases (4894 correct, 1106 incorrect).

| lang | shift on correct | shift on incorrect | kappa (difference) | 95% CI | class-independent? |
|---|---|---|---|---|---|
| de | +0.0004 | +0.0077 | -0.0073 | [-0.0261, +0.0124] | yes |
| es | +0.0284 | +0.0419 | -0.0135 | [-0.0265, +0.0010] | yes |
| fr | -0.0865 | -0.1025 | +0.0160 | [+0.0016, +0.0310] | yes |
| ja | +0.0031 | +0.0364 | -0.0333 | [-0.0506, -0.0166] | **no** |
| vi | +0.0112 | +0.0416 | -0.0304 | [-0.0503, -0.0098] | **no** |
| zh | +0.0029 | +0.0167 | -0.0138 | [-0.0327, +0.0049] | yes |

## qwen7b

6000 candidates over 163 databases (4894 correct, 1106 incorrect).

| lang | shift on correct | shift on incorrect | kappa (difference) | 95% CI | class-independent? |
|---|---|---|---|---|---|
| de | -0.0421 | -0.0481 | +0.0060 | [-0.0328, +0.0463] | yes |
| es | -0.0286 | -0.0312 | +0.0026 | [-0.0375, +0.0439] | yes |
| fr | -0.0238 | -0.0169 | -0.0070 | [-0.0460, +0.0334] | yes |
| ja | -0.0502 | -0.0323 | -0.0179 | [-0.0594, +0.0232] | yes |
| vi | -0.0787 | -0.0904 | +0.0117 | [-0.0397, +0.0606] | yes |
| zh | -0.0594 | -0.0616 | +0.0022 | [-0.0429, +0.0475] | yes |

## Reading this

A small `kappa` supports describing the language effect as an approximately
additive, class-independent score offset **for these graders on this task**.
It is an empirical description, not a causal explanation, and it does not
rule out language-by-item interaction that averages out across items.

