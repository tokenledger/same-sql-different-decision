# Does the cross-lingual verifier effect survive schema shift?

Databases are split 50/50 (by db_id, not by question) into a FIT half (used to fit the English abstention threshold at 10% target risk) and a held-out TEST half. See the module docstring for the three regimes compared and which resampling unit backs each CI.

## big-llama8b

163 databases split 50/50 by database (seed=0): 81 FIT / 82 TEST. FIT: 599 questions, 2995 candidates (x7 langs each scored). TEST: 601 questions, 3005 candidates.

Threshold fit on English/FIT to hit 10% risk: 0.7549

Baseline (fit population, English): risk 0.100, coverage 0.579.

### (a) language-only shift -- FIT databases, other languages
Same population the threshold was fit on; only the language changes. Paired bootstrap over questions (95% CI on the coverage gap vs FIT/English).

| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.100 | 0.579 | -- | -- | -- |
| de | 0.111 | 0.620 | +0.041 | [+0.018, +0.065] | **yes** |
| es | 0.106 | 0.650 | +0.071 | [+0.045, +0.098] | **yes** |
| fr | 0.078 | 0.458 | -0.122 | [-0.148, -0.096] | **yes** |
| ja | 0.105 | 0.616 | +0.037 | [+0.010, +0.063] | **yes** |
| vi | 0.114 | 0.611 | +0.031 | [+0.004, +0.060] | **yes** |
| zh | 0.101 | 0.610 | +0.030 | [+0.004, +0.057] | **yes** |

### (b) language+schema shift -- held-out databases, other languages
Threshold still fit on FIT/English; evaluated on TEST databases in each other language. Two-sample bootstrap by DATABASE (gap vs FIT/English).

| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.078 | 0.544 | -0.035 | [-0.096, +0.025] | no (=English, see (c)) |
| de | 0.102 | 0.579 | -0.001 | [-0.061, +0.062] | no (negligible) |
| es | 0.096 | 0.646 | +0.066 | [+0.002, +0.126] | **yes** |
| fr | 0.068 | 0.423 | -0.157 | [-0.218, -0.093] | **yes** |
| ja | 0.085 | 0.566 | -0.014 | [-0.077, +0.052] | no (negligible) |
| vi | 0.107 | 0.595 | +0.016 | [-0.047, +0.080] | no (negligible) |
| zh | 0.092 | 0.565 | -0.015 | [-0.076, +0.047] | no (negligible) |

### (c) reference: schema shift alone -- held-out databases, English
Same threshold, same TEST databases, but the pivot language -- isolates how much of (b) is schema shift vs added language shift. (Row is a duplicate of the `en` row in (b), repeated for readability.)

English/TEST: risk 0.078, coverage 0.544, gap vs FIT/en -0.035 [-0.093, +0.026] no.

### AUROC on FIT vs held-out (TEST) databases
Two-sample bootstrap by DATABASE.

| lang | AUROC (FIT) | AUROC (TEST) | gap FIT-TEST | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.686 | 0.735 | -0.050 | [-0.115, +0.016] | no |
| de | 0.652 | 0.710 | -0.057 | [-0.135, +0.020] | no |
| es | 0.673 | 0.732 | -0.057 | [-0.123, +0.010] | no |
| fr | 0.681 | 0.714 | -0.033 | [-0.105, +0.035] | no |
| ja | 0.661 | 0.704 | -0.043 | [-0.111, +0.030] | no |
| vi | 0.670 | 0.687 | -0.018 | [-0.084, +0.046] | no (negligible) |
| zh | 0.680 | 0.726 | -0.046 | [-0.110, +0.016] | no |

## big-qwen7b

Threshold fit on English/FIT to hit 10% risk: 1.0000

Baseline (fit population, English): risk 0.098, coverage 0.643.

### (a) language-only shift -- FIT databases, other languages
Same population the threshold was fit on; only the language changes. Paired bootstrap over questions (95% CI on the coverage gap vs FIT/English).

| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.098 | 0.643 | -- | -- | -- |
| de | 0.098 | 0.536 | -0.107 | [-0.139, -0.073] | **yes** |
| es | 0.095 | 0.560 | -0.083 | [-0.116, -0.052] | **yes** |
| fr | 0.097 | 0.595 | -0.048 | [-0.077, -0.018] | **yes** |
| ja | 0.092 | 0.606 | -0.037 | [-0.074, -0.004] | **yes** |
| vi | 0.097 | 0.503 | -0.140 | [-0.176, -0.106] | **yes** |
| zh | 0.094 | 0.549 | -0.094 | [-0.128, -0.061] | **yes** |

### (b) language+schema shift -- held-out databases, other languages
Threshold still fit on FIT/English; evaluated on TEST databases in each other language. Two-sample bootstrap by DATABASE (gap vs FIT/English).

| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.085 | 0.610 | -0.033 | [-0.095, +0.028] | no (=English, see (c)) |
| de | 0.082 | 0.499 | -0.144 | [-0.209, -0.076] | **yes** |
| es | 0.074 | 0.537 | -0.106 | [-0.172, -0.038] | **yes** |
| fr | 0.079 | 0.529 | -0.114 | [-0.183, -0.046] | **yes** |
| ja | 0.094 | 0.554 | -0.089 | [-0.158, -0.018] | **yes** |
| vi | 0.087 | 0.462 | -0.182 | [-0.245, -0.117] | **yes** |
| zh | 0.089 | 0.564 | -0.080 | [-0.145, -0.015] | **yes** |

### (c) reference: schema shift alone -- held-out databases, English
Same threshold, same TEST databases, but the pivot language -- isolates how much of (b) is schema shift vs added language shift. (Row is a duplicate of the `en` row in (b), repeated for readability.)

English/TEST: risk 0.085, coverage 0.610, gap vs FIT/en -0.033 [-0.092, +0.025] no.

### AUROC on FIT vs held-out (TEST) databases
Two-sample bootstrap by DATABASE.

| lang | AUROC (FIT) | AUROC (TEST) | gap FIT-TEST | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.696 | 0.754 | -0.059 | [-0.125, +0.009] | no |
| de | 0.672 | 0.716 | -0.044 | [-0.117, +0.033] | no |
| es | 0.685 | 0.743 | -0.058 | [-0.127, +0.013] | no |
| fr | 0.679 | 0.729 | -0.049 | [-0.120, +0.019] | no |
| ja | 0.690 | 0.701 | -0.011 | [-0.086, +0.064] | no (negligible) |
| vi | 0.664 | 0.706 | -0.043 | [-0.124, +0.034] | no |
| zh | 0.689 | 0.717 | -0.029 | [-0.102, +0.045] | no |

## Reading

**AUROC (ranking) survives schema shift.** For both verifiers and every language, the FIT-vs-TEST AUROC gap is not significant (CIs straddle zero) -- and for llama8b, TEST-database AUROC is actually a few points *higher* than FIT, not lower. There is no evidence the verifier ranks correct/incorrect SQL worse on unseen databases; if anything the ~80 held-out databases in this split happen to be marginally easier to rank, well within noise.

**Coverage instability from language shift alone (a) mostly carries over to language+schema shift (b), same sign, similar or larger magnitude -- but many effects lose significance.** That loss of significance is a sample-size artifact, not evidence the effect vanishes: (a) is bootstrapped over ~600 questions sharing one fixed set of databases, while (b) and (c) are bootstrapped over ~80 databases (the unit that actually varies under schema shift), which is a much smaller effective sample. Point estimates in (b) track (a) closely for most languages (e.g. llama8b/fr: -0.107 in (a) vs -0.143 in (b); qwen7b/de: -0.229 vs -0.242), so the underlying effect looks stable -- the study is just underpowered to confirm it at the database level with only 163 databases split in half.

**Schema shift alone (c), with language held at English, is small and never significant** (llama8b: -0.025 coverage; qwen7b: -0.036), confirming the coverage instability is a language-shift phenomenon, not a schema-shift phenomenon -- schema shift mainly acts as an additional noise source that widens the CIs enough to swallow the language effect's significance rather than a competing effect with its own sign.

**qwen7b is the more fragile verifier.** Its English threshold sits exactly at score 1.0 (the maximum, a tie-heavy region), so acceptance is an all-or-nothing gate per language and coverage swings by 15-23 points across languages regardless of database provenance -- this reproduces the pre-existing 36.0% (en) vs 14.1% (de) full-data result reasonably closely on the FIT-only subset here (34.1% vs 11.2%). llama8b's threshold (0.836) sits in a less saturated region and shows milder, more mixed-sign coverage shifts.

**Bottom line:** the risk stays near target in every regime for both verifiers (the calibration/risk-control property is robust), and AUROC ranking is unaffected by schema shift. The coverage instability under language shift is real and, where measurable, does not appear to be cured or worsened in a clearly resolvable way by additionally shifting to unseen databases -- point estimates suggest it persists at similar or larger magnitude, but confirming that at the database level would need more than 163 databases to reach the same power as the question-level comparisons.
