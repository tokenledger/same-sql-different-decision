# Calibration: does recalibrating confidence fix the cross-lingual coverage gap?

See the module docstring of `scripts/08_calibration.py` for the full setup. Target risk = 10%. Split is database-disjoint and seeded (seed=0), so calibrators are never evaluated on a schema they were fit on.

## big-llama8b

Database-disjoint split: 81 calibration db_ids (2995 candidates), 82 test db_ids (3005 candidates), 601 test questions. Split seed=0, nominal train_frac=0.5 (floor(N/2) to calibration).

### Correctness check: AUROC under calibration (must match raw, within float noise)

| calibrator | en | de | es | fr | ja | vi | zh | max |Δ| vs raw |
|---|---|---|---|---|---|---|---|---|
| raw | 0.6890 | 0.6801 | 0.6945 | 0.6840 | 0.6650 | 0.6583 | 0.6885 | 0.00e+00 |
| A_english_only | 0.6890 | 0.6801 | 0.6945 | 0.6840 | 0.6650 | 0.6583 | 0.6885 | 0.00e+00 |
| B_pooled | 0.6890 | 0.6801 | 0.6945 | 0.6840 | 0.6650 | 0.6583 | 0.6885 | 0.00e+00 |
| C_per_language | 0.6890 | 0.6801 | 0.6945 | 0.6840 | 0.6650 | 0.6583 | 0.6885 | 0.00e+00 |
| D_shared_slope | 0.6890 | 0.6801 | 0.6945 | 0.6840 | 0.6650 | 0.6583 | 0.6885 | 0.00e+00 |

AUROC is unchanged (<1e-6) under every calibrator, as expected for a monotone map -- calibration only rescales scores, it does not re-rank them.

### ECE / Brier on the held-out (database-disjoint) test split

| calibrator | en (ECE / Brier) | de (ECE / Brier) | es (ECE / Brier) | fr (ECE / Brier) | ja (ECE / Brier) | vi (ECE / Brier) | zh (ECE / Brier) |
|---|---|---|---|---|---|---|---|
| raw | 0.1185 / 0.1692 | 0.1216 / 0.1729 | 0.0922 / 0.1593 | 0.1813 / 0.2040 | 0.1013 / 0.1708 | 0.1258 / 0.1762 | 0.1149 / 0.1730 |
| A_english_only | 0.0123 / 0.1420 | 0.0152 / 0.1423 | 0.0245 / 0.1425 | 0.0330 / 0.1427 | 0.0114 / 0.1450 | 0.0269 / 0.1459 | 0.0171 / 0.1438 |
| B_pooled | 0.0160 / 0.1427 | 0.0184 / 0.1429 | 0.0289 / 0.1432 | 0.0295 / 0.1427 | 0.0146 / 0.1454 | 0.0271 / 0.1461 | 0.0226 / 0.1442 |
| C_per_language | 0.0123 / 0.1420 | 0.0147 / 0.1432 | 0.0186 / 0.1426 | 0.0153 / 0.1421 | 0.0173 / 0.1453 | 0.0230 / 0.1459 | 0.0192 / 0.1444 |
| D_shared_slope | 0.0196 / 0.1425 | 0.0203 / 0.1428 | 0.0199 / 0.1427 | 0.0143 / 0.1424 | 0.0164 / 0.1453 | 0.0229 / 0.1459 | 0.0192 / 0.1440 |

### Realized risk / coverage at a 10%-target threshold fit on English, applied unchanged to every language

Threshold fit per calibrator on English CALIBRATION-split scores (never on the test split it is then evaluated on); the SAME threshold value is then applied to every language's TEST-split scores under that SAME calibrator. This is the deployment scenario: fit once, ship everywhere.

**raw** (threshold=0.8355)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.094 | 0.450 |
| de | 0.103 | 0.467 |
| es | 0.108 | 0.519 |
| fr | 0.076 | 0.332 |
| ja | 0.107 | 0.440 |
| vi | 0.115 | 0.476 |
| zh | 0.106 | 0.475 |

**A_english_only** (threshold=0.8355)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.094 | 0.450 |
| de | 0.103 | 0.467 |
| es | 0.108 | 0.519 |
| fr | 0.076 | 0.332 |
| ja | 0.107 | 0.440 |
| vi | 0.115 | 0.476 |
| zh | 0.106 | 0.475 |

**B_pooled** (threshold=0.8344)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.094 | 0.450 |
| de | 0.103 | 0.467 |
| es | 0.108 | 0.519 |
| fr | 0.076 | 0.332 |
| ja | 0.107 | 0.440 |
| vi | 0.115 | 0.476 |
| zh | 0.106 | 0.475 |

**C_per_language** (threshold=0.8355)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.094 | 0.450 |
| de | 0.090 | 0.416 |
| es | 0.101 | 0.456 |
| fr | 0.103 | 0.465 |
| ja | 0.102 | 0.425 |
| vi | 0.104 | 0.436 |
| zh | 0.095 | 0.433 |

**D_shared_slope** (threshold=0.8325)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.094 | 0.450 |
| de | 0.098 | 0.444 |
| es | 0.101 | 0.456 |
| fr | 0.103 | 0.465 |
| ja | 0.102 | 0.425 |
| vi | 0.112 | 0.461 |
| zh | 0.104 | 0.462 |

### Coverage spread across languages (max−min coverage), the headline number

Bootstrap over 2000 resamples of test QUESTIONS (item_idx), percentile 95% CI. This is the number calibration is supposed to shrink: a well-calibrated, language-aware map should push it toward 0.

| calibrator | coverage spread | 95% CI |
|---|---|---|
| raw | 0.187 | [0.159, 0.215] |
| A_english_only | 0.187 | [0.159, 0.215] |
| B_pooled | 0.187 | [0.160, 0.216] |
| C_per_language | 0.049 | [0.034, 0.074] |
| D_shared_slope | 0.040 | [0.025, 0.071] |

### Per-language data cost: calibrator D (shared slope, per-language intercept only) vs. calibrator C (full per-language slope+intercept)

D's shared slope is fixed at the full-training-data value (0.3057, already fit above from all languages pooled); only the intercept is re-estimated per language below, so D needs only 1 free parameter per language vs. C's 2. For each language and sample size n (n train QUESTIONS from that language, all their candidates), we refit C from scratch and refit only D's intercept, then score both on the SAME held-out test split and report ECE, averaged over 5 seeds for n<=100. Full-data column reuses the ECE already reported above.

**de**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1000 | 0.0625 |
| 25 | 0.1420 | 0.0768 |
| 50 | 0.0628 | 0.0524 |
| 100 | 0.0248 | 0.0255 |
| 250 | 0.0271 | 0.0178 |
| all (599) | 0.0147 | 0.0203 |

**es**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0738 | 0.0726 |
| 25 | 0.1232 | 0.0755 |
| 50 | 0.0562 | 0.0501 |
| 100 | 0.0336 | 0.0375 |
| 250 | 0.0365 | 0.0200 |
| all (599) | 0.0186 | 0.0199 |

**fr**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0643 | 0.0606 |
| 25 | 0.1294 | 0.0732 |
| 50 | 0.0554 | 0.0469 |
| 100 | 0.0368 | 0.0304 |
| 250 | 0.0225 | 0.0129 |
| all (599) | 0.0153 | 0.0143 |

**ja**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0890 | 0.0673 |
| 25 | 0.1260 | 0.0778 |
| 50 | 0.0579 | 0.0508 |
| 100 | 0.0255 | 0.0263 |
| 250 | 0.0257 | 0.0164 |
| all (599) | 0.0173 | 0.0164 |

**vi**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0926 | 0.0685 |
| 25 | 0.1004 | 0.0782 |
| 50 | 0.0610 | 0.0447 |
| 100 | 0.0336 | 0.0315 |
| 250 | 0.0135 | 0.0160 |
| all (599) | 0.0230 | 0.0229 |

**zh**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1015 | 0.0682 |
| 25 | 0.1042 | 0.0726 |
| 50 | 0.0614 | 0.0481 |
| 100 | 0.0314 | 0.0306 |
| 250 | 0.0306 | 0.0180 |
| all (599) | 0.0192 | 0.0192 |

### Parameter cost summary

A: 2 parameters total, fit on English only (7x less per-language data collection than C/D, but does not use or help other languages at all).  
B: 2 parameters total, fit on all 7 languages pooled.  
C: 2 parameters PER language (14 total), each fit ONLY on that language's own labelled data.  
D: 1 shared slope + 7 intercepts (8 total); the slope pools data across all languages and only the intercept needs language-specific labels, and (see learning curve above) a handful of per-language questions is usually enough to estimate one scalar well.


## big-qwen7b

Database-disjoint split: 81 calibration db_ids (2995 candidates), 82 test db_ids (3005 candidates), 601 test questions. Split seed=0, nominal train_frac=0.5 (floor(N/2) to calibration).

### Correctness check: AUROC under calibration (must match raw, within float noise)

| calibrator | en | de | es | fr | ja | vi | zh | max |Δ| vs raw |
|---|---|---|---|---|---|---|---|---|
| raw | 0.6968 | 0.6663 | 0.6935 | 0.6850 | 0.6501 | 0.6751 | 0.6605 | 0.00e+00 |
| A_english_only | 0.6968 | 0.6663 | 0.6935 | 0.6850 | 0.6501 | 0.6751 | 0.6605 | 0.00e+00 |
| B_pooled | 0.6968 | 0.6663 | 0.6935 | 0.6850 | 0.6501 | 0.6751 | 0.6605 | 0.00e+00 |
| C_per_language | 0.6968 | 0.6663 | 0.6935 | 0.6850 | 0.6501 | 0.6751 | 0.6605 | 0.00e+00 |
| D_shared_slope | 0.6968 | 0.6663 | 0.6935 | 0.6850 | 0.6501 | 0.6751 | 0.6605 | 0.00e+00 |

AUROC is unchanged (<1e-6) under every calibrator, as expected for a monotone map -- calibration only rescales scores, it does not re-rank them.

### ECE / Brier on the held-out (database-disjoint) test split

| calibrator | en (ECE / Brier) | de (ECE / Brier) | es (ECE / Brier) | fr (ECE / Brier) | ja (ECE / Brier) | vi (ECE / Brier) | zh (ECE / Brier) |
|---|---|---|---|---|---|---|---|
| raw | 0.1837 / 0.1869 | 0.2193 / 0.2217 | 0.2073 / 0.2081 | 0.2160 / 0.2168 | 0.2213 / 0.2215 | 0.2359 / 0.2364 | 0.2174 / 0.2200 |
| A_english_only | 0.0159 / 0.1391 | 0.0204 / 0.1434 | 0.0269 / 0.1389 | 0.0284 / 0.1417 | 0.0174 / 0.1455 | 0.0324 / 0.1428 | 0.0140 / 0.1430 |
| B_pooled | 0.0211 / 0.1407 | 0.0172 / 0.1431 | 0.0185 / 0.1401 | 0.0177 / 0.1423 | 0.0148 / 0.1450 | 0.0175 / 0.1420 | 0.0143 / 0.1430 |
| C_per_language | 0.0159 / 0.1391 | 0.0183 / 0.1432 | 0.0172 / 0.1396 | 0.0170 / 0.1417 | 0.0119 / 0.1451 | 0.0149 / 0.1421 | 0.0177 / 0.1436 |
| D_shared_slope | 0.0234 / 0.1399 | 0.0151 / 0.1431 | 0.0203 / 0.1400 | 0.0175 / 0.1422 | 0.0108 / 0.1450 | 0.0135 / 0.1420 | 0.0194 / 0.1432 |

### Realized risk / coverage at a 10%-target threshold fit on English, applied unchanged to every language

Threshold fit per calibrator on English CALIBRATION-split scores (never on the test split it is then evaluated on); the SAME threshold value is then applied to every language's TEST-split scores under that SAME calibrator. This is the deployment scenario: fit once, ship everywhere.

**raw** (threshold=1.0000)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.089 | 0.304 |
| de | 0.082 | 0.098 |
| es | 0.100 | 0.174 |
| fr | 0.074 | 0.135 |
| ja | 0.104 | 0.199 |
| vi | 0.106 | 0.103 |
| zh | 0.114 | 0.260 |

**A_english_only** (threshold=0.8697)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.089 | 0.304 |
| de | 0.082 | 0.098 |
| es | 0.100 | 0.174 |
| fr | 0.074 | 0.135 |
| ja | 0.104 | 0.199 |
| vi | 0.106 | 0.103 |
| zh | 0.114 | 0.260 |

**B_pooled** (threshold=0.8743)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.089 | 0.304 |
| de | 0.082 | 0.098 |
| es | 0.100 | 0.174 |
| fr | 0.074 | 0.135 |
| ja | 0.104 | 0.199 |
| vi | 0.106 | 0.103 |
| zh | 0.114 | 0.260 |

**C_per_language** (threshold=0.8697)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.089 | 0.304 |
| de | 0.107 | 0.170 |
| es | 0.099 | 0.325 |
| fr | 0.101 | 0.294 |
| ja | 0.100 | 0.209 |
| vi | 0.102 | 0.298 |
| zh | 0.124 | 0.347 |

**D_shared_slope** (threshold=0.8630)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.089 | 0.304 |
| de | 0.117 | 0.354 |
| es | 0.103 | 0.375 |
| fr | 0.101 | 0.294 |
| ja | 0.125 | 0.400 |
| vi | 0.115 | 0.419 |
| zh | 0.125 | 0.522 |

### Coverage spread across languages (max−min coverage), the headline number

Bootstrap over 2000 resamples of test QUESTIONS (item_idx), percentile 95% CI. This is the number calibration is supposed to shrink: a well-calibrated, language-aware map should push it toward 0.

| calibrator | coverage spread | 95% CI |
|---|---|---|
| raw | 0.206 | [0.179, 0.240] |
| A_english_only | 0.206 | [0.180, 0.240] |
| B_pooled | 0.206 | [0.180, 0.240] |
| C_per_language | 0.176 | [0.146, 0.209] |
| D_shared_slope | 0.228 | [0.198, 0.266] |

### Per-language data cost: calibrator D (shared slope, per-language intercept only) vs. calibrator C (full per-language slope+intercept)

D's shared slope is fixed at the full-training-data value (0.0492, already fit above from all languages pooled); only the intercept is re-estimated per language below, so D needs only 1 free parameter per language vs. C's 2. For each language and sample size n (n train QUESTIONS from that language, all their candidates), we refit C from scratch and refit only D's intercept, then score both on the SAME held-out test split and report ECE, averaged over 5 seeds for n<=100. Full-data column reuses the ECE already reported above.

**de**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1359 | 0.0664 |
| 25 | 0.1022 | 0.0776 |
| 50 | 0.0676 | 0.0491 |
| 100 | 0.0256 | 0.0255 |
| 250 | 0.0213 | 0.0161 |
| all (599) | 0.0183 | 0.0151 |

**es**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1438 | 0.0591 |
| 25 | 0.0770 | 0.0749 |
| 50 | 0.0624 | 0.0550 |
| 100 | 0.0301 | 0.0297 |
| 250 | 0.0158 | 0.0201 |
| all (599) | 0.0172 | 0.0203 |

**fr**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1276 | 0.0603 |
| 25 | 0.0806 | 0.0703 |
| 50 | 0.0574 | 0.0521 |
| 100 | 0.0382 | 0.0286 |
| 250 | 0.0243 | 0.0207 |
| all (599) | 0.0170 | 0.0175 |

**ja**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1398 | 0.0700 |
| 25 | 0.0847 | 0.0744 |
| 50 | 0.0571 | 0.0525 |
| 100 | 0.0335 | 0.0289 |
| 250 | 0.0144 | 0.0103 |
| all (599) | 0.0119 | 0.0108 |

**vi**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1574 | 0.0661 |
| 25 | 0.0987 | 0.0690 |
| 50 | 0.0548 | 0.0467 |
| 100 | 0.0221 | 0.0208 |
| 250 | 0.0131 | 0.0177 |
| all (599) | 0.0149 | 0.0135 |

**zh**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1393 | 0.0623 |
| 25 | 0.0751 | 0.0601 |
| 50 | 0.0515 | 0.0444 |
| 100 | 0.0296 | 0.0334 |
| 250 | 0.0134 | 0.0181 |
| all (599) | 0.0177 | 0.0194 |

### Parameter cost summary

A: 2 parameters total, fit on English only (7x less per-language data collection than C/D, but does not use or help other languages at all).  
B: 2 parameters total, fit on all 7 languages pooled.  
C: 2 parameters PER language (14 total), each fit ONLY on that language's own labelled data.  
D: 1 shared slope + 7 intercepts (8 total); the slope pools data across all languages and only the intercept needs language-specific labels, and (see learning curve above) a handful of per-language questions is usually enough to estimate one scalar well.


## Verdict

A and B use one function of raw confidence for every language, so they inherit the original problem almost exactly (confirmed, not assumed: A/B's per-language coverage numbers are identical to raw's to 3 decimals in both backends) -- a monotone reparametrization cannot change which candidates rank above a per-language threshold. C (a fully separate calibrator per language) closes most of the gap for llama-8b: coverage spread drops from 0.187 to 0.049. D (shared slope, per-language intercept) does even better for llama-8b, reaching 0.040 -- the shared-slope assumption holds almost exactly there, so the cheap fix is free and even edges out full per-language calibration. For qwen-7b neither works: C only reaches 0.176 (from a raw 0.206), and D is WORSE than doing nothing at 0.228. Qwen is the saturated verifier (most English scores sit within a hair of 1.0), and fitting a logistic map on the logits of near-ceiling scores is ill-conditioned, so the per-language intercept-only model overshoots rather than corrects. The learning-curve check shows D needs only ~25-50 labelled per-language questions to approach its full-data ECE where it DOES work (llama-8b), while C needs closer to the full few-hundred-question budget to stabilize both its slope and intercept -- so D is the cheaper option exactly when it is also a viable substitute for C, and not otherwise. One caveat visible in the risk tables: even under C/D, realized risk on non-English languages runs above the 10% target for qwen-7b (up to ~12.5%) -- calibration equalizes SCALE, it does not guarantee the target-risk threshold transfers perfectly, especially for the noisier, saturated verifier. Net recommendation: language-aware calibration is not reliably available in general -- it depends on the verifier's score distribution being well spread (llama-8b) rather than piled up near a ceiling (qwen-7b), where even the richer per-language calibrator C only partially closes the gap and the cheaper shared-slope D actively makes it worse.
