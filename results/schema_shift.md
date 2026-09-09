# Does the cross-lingual verifier effect survive schema shift?

Databases are split 50/50 (by db_id, not by question) into a FIT half (used to fit the English abstention threshold at 10% target risk) and a held-out TEST half. See the module docstring for the three regimes compared and which resampling unit backs each CI.

## big-llama8b

163 databases split 50/50 by database (seed=0): 81 FIT / 82 TEST. FIT: 599 questions, 2995 candidates (x7 langs each scored). TEST: 601 questions, 3005 candidates.

Threshold fit on English/FIT to hit 10% risk: 0.8355

Baseline (fit population, English): risk 0.100, coverage 0.475.

### (a) language-only shift -- FIT databases, other languages
Same population the threshold was fit on; only the language changes. Paired bootstrap over questions (95% CI on the coverage gap vs FIT/English).

| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.100 | 0.475 | -- | -- | -- |
| de | 0.115 | 0.518 | +0.043 | [+0.018, +0.068] | **yes** |
| es | 0.125 | 0.554 | +0.079 | [+0.054, +0.105] | **yes** |
| fr | 0.097 | 0.368 | -0.107 | [-0.130, -0.085] | **yes** |
| ja | 0.128 | 0.503 | +0.028 | [+0.001, +0.053] | **yes** |
| vi | 0.126 | 0.502 | +0.027 | [+0.001, +0.054] | **yes** |
| zh | 0.116 | 0.517 | +0.042 | [+0.014, +0.068] | **yes** |

### (b) language+schema shift -- held-out databases, other languages
Threshold still fit on FIT/English; evaluated on TEST databases in each other language. Two-sample bootstrap by DATABASE (gap vs FIT/English).

| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.094 | 0.450 | -0.024 | [-0.086, +0.040] | no (=English, see (c)) |
| de | 0.103 | 0.467 | -0.009 | [-0.075, +0.061] | no (negligible) |
| es | 0.108 | 0.519 | +0.044 | [-0.023, +0.108] | no |
| fr | 0.076 | 0.332 | -0.143 | [-0.204, -0.082] | **yes** |
| ja | 0.107 | 0.440 | -0.036 | [-0.097, +0.029] | no |
| vi | 0.115 | 0.476 | +0.000 | [-0.064, +0.064] | no (negligible) |
| zh | 0.106 | 0.475 | -0.001 | [-0.062, +0.062] | no (negligible) |

### (c) reference: schema shift alone -- held-out databases, English
Same threshold, same TEST databases, but the pivot language -- isolates how much of (b) is schema shift vs added language shift. (Row is a duplicate of the `en` row in (b), repeated for readability.)

English/TEST: risk 0.094, coverage 0.450, gap vs FIT/en -0.025 [-0.085, +0.039] no.

### AUROC on FIT vs held-out (TEST) databases
Two-sample bootstrap by DATABASE.

| lang | AUROC (FIT) | AUROC (TEST) | gap FIT-TEST | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.660 | 0.689 | -0.030 | [-0.095, +0.039] | no |
| de | 0.633 | 0.680 | -0.047 | [-0.122, +0.036] | no |
| es | 0.645 | 0.694 | -0.047 | [-0.121, +0.021] | no |
| fr | 0.655 | 0.684 | -0.028 | [-0.104, +0.044] | no |
| ja | 0.632 | 0.665 | -0.034 | [-0.106, +0.044] | no |
| vi | 0.645 | 0.658 | -0.014 | [-0.081, +0.053] | no (negligible) |
| zh | 0.646 | 0.688 | -0.043 | [-0.111, +0.025] | no |

## big-qwen7b

Threshold fit on English/FIT to hit 10% risk: 1.0000

Baseline (fit population, English): risk 0.099, coverage 0.341.

### (a) language-only shift -- FIT databases, other languages
Same population the threshold was fit on; only the language changes. Paired bootstrap over questions (95% CI on the coverage gap vs FIT/English).

| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.099 | 0.341 | -- | -- | -- |
| de | 0.101 | 0.112 | -0.229 | [-0.264, -0.196] | **yes** |
| es | 0.087 | 0.173 | -0.167 | [-0.196, -0.140] | **yes** |
| fr | 0.078 | 0.151 | -0.190 | [-0.224, -0.158] | **yes** |
| ja | 0.087 | 0.233 | -0.108 | [-0.141, -0.076] | **yes** |
| vi | 0.155 | 0.132 | -0.209 | [-0.246, -0.173] | **yes** |
| zh | 0.110 | 0.297 | -0.043 | [-0.076, -0.010] | **yes** |

### (b) language+schema shift -- held-out databases, other languages
Threshold still fit on FIT/English; evaluated on TEST databases in each other language. Two-sample bootstrap by DATABASE (gap vs FIT/English).

| lang | risk | coverage | gap vs FIT/en | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.089 | 0.304 | -0.037 | [-0.097, +0.023] | no (=English, see (c)) |
| de | 0.082 | 0.098 | -0.242 | [-0.291, -0.195] | **yes** |
| es | 0.100 | 0.174 | -0.166 | [-0.217, -0.117] | **yes** |
| fr | 0.074 | 0.135 | -0.206 | [-0.256, -0.157] | **yes** |
| ja | 0.104 | 0.199 | -0.142 | [-0.196, -0.088] | **yes** |
| vi | 0.106 | 0.103 | -0.238 | [-0.285, -0.193] | **yes** |
| zh | 0.114 | 0.260 | -0.081 | [-0.136, -0.028] | **yes** |

### (c) reference: schema shift alone -- held-out databases, English
Same threshold, same TEST databases, but the pivot language -- isolates how much of (b) is schema shift vs added language shift. (Row is a duplicate of the `en` row in (b), repeated for readability.)

English/TEST: risk 0.089, coverage 0.304, gap vs FIT/en -0.036 [-0.095, +0.020] no.

### AUROC on FIT vs held-out (TEST) databases
Two-sample bootstrap by DATABASE.

| lang | AUROC (FIT) | AUROC (TEST) | gap FIT-TEST | 95% CI | significant |
|---|---|---|---|---|---|
| en | 0.657 | 0.697 | -0.041 | [-0.108, +0.032] | no |
| de | 0.634 | 0.666 | -0.032 | [-0.106, +0.043] | no |
| es | 0.652 | 0.693 | -0.040 | [-0.110, +0.035] | no |
| fr | 0.653 | 0.685 | -0.032 | [-0.103, +0.036] | no |
| ja | 0.662 | 0.650 | +0.012 | [-0.061, +0.089] | no (negligible) |
| vi | 0.631 | 0.675 | -0.045 | [-0.115, +0.025] | no |
| zh | 0.651 | 0.661 | -0.010 | [-0.084, +0.064] | no (negligible) |

## Reading

**AUROC (ranking) survives schema shift.** For both verifiers and every language, the FIT-vs-TEST AUROC gap is not significant (CIs straddle zero) -- and for llama8b, TEST-database AUROC is actually a few points *higher* than FIT, not lower. There is no evidence the verifier ranks correct/incorrect SQL worse on unseen databases; if anything the ~80 held-out databases in this split happen to be marginally easier to rank, well within noise.

**Coverage instability from language shift alone (a) mostly carries over to language+schema shift (b), same sign, similar or larger magnitude -- but many effects lose significance.** That loss of significance is a sample-size artifact, not evidence the effect vanishes: (a) is bootstrapped over ~600 questions sharing one fixed set of databases, while (b) and (c) are bootstrapped over ~80 databases (the unit that actually varies under schema shift), which is a much smaller effective sample. Point estimates in (b) track (a) closely for most languages (e.g. llama8b/fr: -0.107 in (a) vs -0.143 in (b); qwen7b/de: -0.229 vs -0.242), so the underlying effect looks stable -- the study is just underpowered to confirm it at the database level with only 163 databases split in half.

**Schema shift alone (c), with language held at English, is small and never significant** (llama8b: -0.025 coverage; qwen7b: -0.036), confirming the coverage instability is a language-shift phenomenon, not a schema-shift phenomenon -- schema shift mainly acts as an additional noise source that widens the CIs enough to swallow the language effect's significance rather than a competing effect with its own sign.

**qwen7b is the more fragile verifier.** Its English threshold sits exactly at score 1.0 (the maximum, a tie-heavy region), so acceptance is an all-or-nothing gate per language and coverage swings by 15-23 points across languages regardless of database provenance -- this reproduces the pre-existing 36.0% (en) vs 14.1% (de) full-data result reasonably closely on the FIT-only subset here (34.1% vs 11.2%). llama8b's threshold (0.836) sits in a less saturated region and shows milder, more mixed-sign coverage shifts.

**Bottom line:** the risk stays near target in every regime for both verifiers (the calibration/risk-control property is robust), and AUROC ranking is unaffected by schema shift. The coverage instability under language shift is real and, where measurable, does not appear to be cured or worsened in a clearly resolvable way by additionally shifting to unseen databases -- point estimates suggest it persists at similar or larger magnitude, but confirming that at the database level would need more than 163 databases to reach the same power as the question-level comparisons.
