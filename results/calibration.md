# Calibration: does recalibrating confidence fix the cross-lingual coverage gap?

See the module docstring of `scripts/08_calibration.py` for the full setup. Target risk = 10%. Split is database-disjoint and seeded (seed=0), so calibrators are never evaluated on a schema they were fit on.

## big-llama8b

Database-disjoint split: 81 calibration db_ids (2995 candidates), 82 test db_ids (3005 candidates), 601 test questions. Split seed=0, nominal train_frac=0.5 (floor(N/2) to calibration).

### Correctness check: AUROC under calibration (must match raw, within float noise)

| calibrator | en | de | es | fr | ja | vi | zh | max |Δ| vs raw |
|---|---|---|---|---|---|---|---|---|
| raw | 0.7346 | 0.7099 | 0.7318 | 0.7140 | 0.7035 | 0.6866 | 0.7256 | 0.00e+00 |
| A_english_only | 0.7346 | 0.7099 | 0.7318 | 0.7140 | 0.7035 | 0.6866 | 0.7256 | 0.00e+00 |
| B_pooled | 0.7346 | 0.7099 | 0.7318 | 0.7140 | 0.7035 | 0.6866 | 0.7256 | 0.00e+00 |
| C_per_language | 0.7346 | 0.7099 | 0.7318 | 0.7140 | 0.7035 | 0.6866 | 0.7256 | 0.00e+00 |
| D_shared_slope | 0.7346 | 0.7099 | 0.7318 | 0.7140 | 0.7035 | 0.6866 | 0.7256 | 0.00e+00 |

AUROC is unchanged (<1e-6) under every calibrator, as expected for a monotone map -- calibration only rescales scores, it does not re-rank them.

### ECE / Brier on the held-out (database-disjoint) test split

| calibrator | en (ECE / Brier) | de (ECE / Brier) | es (ECE / Brier) | fr (ECE / Brier) | ja (ECE / Brier) | vi (ECE / Brier) | zh (ECE / Brier) |
|---|---|---|---|---|---|---|---|
| raw | 0.1167 / 0.1514 | 0.1179 / 0.1598 | 0.0849 / 0.1414 | 0.2022 / 0.1944 | 0.1054 / 0.1544 | 0.1221 / 0.1607 | 0.1173 / 0.1561 |
| A_english_only | 0.0179 / 0.1221 | 0.0314 / 0.1241 | 0.0337 / 0.1234 | 0.0382 / 0.1239 | 0.0188 / 0.1259 | 0.0358 / 0.1272 | 0.0262 / 0.1246 |
| B_pooled | 0.0280 / 0.1229 | 0.0226 / 0.1246 | 0.0401 / 0.1241 | 0.0385 / 0.1238 | 0.0233 / 0.1263 | 0.0310 / 0.1274 | 0.0295 / 0.1250 |
| C_per_language | 0.0179 / 0.1221 | 0.0178 / 0.1250 | 0.0285 / 0.1233 | 0.0205 / 0.1232 | 0.0188 / 0.1261 | 0.0360 / 0.1272 | 0.0248 / 0.1251 |
| D_shared_slope | 0.0248 / 0.1226 | 0.0219 / 0.1244 | 0.0296 / 0.1235 | 0.0188 / 0.1234 | 0.0188 / 0.1261 | 0.0370 / 0.1272 | 0.0265 / 0.1249 |

### Realized risk / coverage at a 10%-target threshold fit on English, applied unchanged to every language

Threshold fit per calibrator on English CALIBRATION-split scores (never on the test split it is then evaluated on); the SAME threshold value is then applied to every language's TEST-split scores under that SAME calibrator. This is the deployment scenario: fit once, ship everywhere.

**raw** (threshold=0.7549)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.078 | 0.544 |
| de | 0.102 | 0.579 |
| es | 0.096 | 0.646 |
| fr | 0.068 | 0.423 |
| ja | 0.085 | 0.566 |
| vi | 0.107 | 0.595 |
| zh | 0.092 | 0.565 |

**A_english_only** (threshold=0.8400)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.078 | 0.544 |
| de | 0.102 | 0.579 |
| es | 0.096 | 0.646 |
| fr | 0.068 | 0.423 |
| ja | 0.085 | 0.566 |
| vi | 0.107 | 0.595 |
| zh | 0.092 | 0.565 |

**B_pooled** (threshold=0.8413)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.078 | 0.544 |
| de | 0.102 | 0.579 |
| es | 0.096 | 0.646 |
| fr | 0.068 | 0.423 |
| ja | 0.085 | 0.566 |
| vi | 0.107 | 0.595 |
| zh | 0.092 | 0.565 |

**C_per_language** (threshold=0.8400)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.078 | 0.544 |
| de | 0.099 | 0.556 |
| es | 0.088 | 0.579 |
| fr | 0.094 | 0.579 |
| ja | 0.083 | 0.542 |
| vi | 0.106 | 0.583 |
| zh | 0.089 | 0.556 |

**D_shared_slope** (threshold=0.8389)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.078 | 0.544 |
| de | 0.099 | 0.556 |
| es | 0.088 | 0.579 |
| fr | 0.094 | 0.579 |
| ja | 0.083 | 0.542 |
| vi | 0.106 | 0.583 |
| zh | 0.089 | 0.556 |

### Coverage spread across languages (max−min coverage), the headline number

Bootstrap over 2000 resamples of test QUESTIONS (item_idx), percentile 95% CI. This is the number calibration is supposed to shrink: a well-calibrated, language-aware map should push it toward 0.

| calibrator | coverage spread | 95% CI |
|---|---|---|
| raw | 0.223 | [0.193, 0.253] |
| A_english_only | 0.223 | [0.193, 0.253] |
| B_pooled | 0.223 | [0.193, 0.253] |
| C_per_language | 0.040 | [0.031, 0.077] |
| D_shared_slope | 0.040 | [0.031, 0.078] |

### Per-language data cost: calibrator D (shared slope, per-language intercept only) vs. calibrator C (full per-language slope+intercept)

D's shared slope is fixed at the full-training-data value (0.3623, already fit above from all languages pooled); only the intercept is re-estimated per language below, so D needs only 1 free parameter per language vs. C's 2. For each language and sample size n (n train QUESTIONS from that language, all their candidates), we refit C from scratch and refit only D's intercept, then score both on the SAME held-out test split and report ECE, averaged over 5 seeds for n<=100. Full-data column reuses the ECE already reported above.

**de**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0848 | 0.0470 |
| 25 | 0.1237 | 0.0620 |
| 50 | 0.0672 | 0.0588 |
| 100 | 0.0331 | 0.0338 |
| 250 | 0.0303 | 0.0300 |
| all (599) | 0.0178 | 0.0219 |

**es**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0478 | 0.0559 |
| 25 | 0.1088 | 0.0584 |
| 50 | 0.0585 | 0.0575 |
| 100 | 0.0355 | 0.0359 |
| 250 | 0.0418 | 0.0384 |
| all (599) | 0.0285 | 0.0296 |

**fr**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0435 | 0.0439 |
| 25 | 0.1146 | 0.0563 |
| 50 | 0.0539 | 0.0511 |
| 100 | 0.0339 | 0.0321 |
| 250 | 0.0296 | 0.0313 |
| all (599) | 0.0205 | 0.0188 |

**ja**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0889 | 0.0506 |
| 25 | 0.1106 | 0.0632 |
| 50 | 0.0563 | 0.0572 |
| 100 | 0.0311 | 0.0293 |
| 250 | 0.0254 | 0.0265 |
| all (599) | 0.0188 | 0.0188 |

**vi**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0721 | 0.0536 |
| 25 | 0.0860 | 0.0618 |
| 50 | 0.0665 | 0.0525 |
| 100 | 0.0398 | 0.0340 |
| 250 | 0.0345 | 0.0313 |
| all (599) | 0.0360 | 0.0370 |

**zh**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0873 | 0.0493 |
| 25 | 0.0850 | 0.0589 |
| 50 | 0.0586 | 0.0573 |
| 100 | 0.0409 | 0.0360 |
| 250 | 0.0311 | 0.0314 |
| all (599) | 0.0248 | 0.0265 |

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
| raw | 0.7540 | 0.7158 | 0.7427 | 0.7286 | 0.7005 | 0.7057 | 0.7174 | 0.00e+00 |
| A_english_only | 0.7540 | 0.7158 | 0.7427 | 0.7286 | 0.7005 | 0.7057 | 0.7174 | 0.00e+00 |
| B_pooled | 0.7540 | 0.7158 | 0.7427 | 0.7286 | 0.7005 | 0.7057 | 0.7174 | 0.00e+00 |
| C_per_language | 0.7540 | 0.7158 | 0.7427 | 0.7286 | 0.7005 | 0.7057 | 0.7174 | 0.00e+00 |
| D_shared_slope | 0.7540 | 0.7158 | 0.7427 | 0.7286 | 0.7005 | 0.7057 | 0.7174 | 0.00e+00 |

AUROC is unchanged (<1e-6) under every calibrator, as expected for a monotone map -- calibration only rescales scores, it does not re-rank them.

### ECE / Brier on the held-out (database-disjoint) test split

| calibrator | en (ECE / Brier) | de (ECE / Brier) | es (ECE / Brier) | fr (ECE / Brier) | ja (ECE / Brier) | vi (ECE / Brier) | zh (ECE / Brier) |
|---|---|---|---|---|---|---|---|
| raw | 0.1574 / 0.1602 | 0.1957 / 0.1981 | 0.1803 / 0.1812 | 0.1890 / 0.1902 | 0.1956 / 0.1960 | 0.2183 / 0.2171 | 0.1918 / 0.1939 |
| A_english_only | 0.0273 / 0.1186 | 0.0279 / 0.1241 | 0.0285 / 0.1182 | 0.0237 / 0.1216 | 0.0227 / 0.1254 | 0.0359 / 0.1243 | 0.0239 / 0.1224 |
| B_pooled | 0.0493 / 0.1206 | 0.0225 / 0.1237 | 0.0228 / 0.1199 | 0.0211 / 0.1224 | 0.0231 / 0.1248 | 0.0187 / 0.1230 | 0.0284 / 0.1225 |
| C_per_language | 0.0273 / 0.1186 | 0.0240 / 0.1237 | 0.0210 / 0.1192 | 0.0170 / 0.1218 | 0.0202 / 0.1250 | 0.0132 / 0.1232 | 0.0299 / 0.1232 |
| D_shared_slope | 0.0321 / 0.1197 | 0.0215 / 0.1236 | 0.0189 / 0.1197 | 0.0162 / 0.1222 | 0.0198 / 0.1248 | 0.0159 / 0.1230 | 0.0298 / 0.1227 |

### Realized risk / coverage at a 10%-target threshold fit on English, applied unchanged to every language

Threshold fit per calibrator on English CALIBRATION-split scores (never on the test split it is then evaluated on); the SAME threshold value is then applied to every language's TEST-split scores under that SAME calibrator. This is the deployment scenario: fit once, ship everywhere.

**raw** (threshold=1.0000)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.085 | 0.610 |
| de | 0.082 | 0.499 |
| es | 0.074 | 0.537 |
| fr | 0.079 | 0.529 |
| ja | 0.094 | 0.554 |
| vi | 0.087 | 0.462 |
| zh | 0.089 | 0.564 |

**A_english_only** (threshold=0.8689)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.085 | 0.610 |
| de | 0.082 | 0.499 |
| es | 0.074 | 0.537 |
| fr | 0.079 | 0.529 |
| ja | 0.094 | 0.554 |
| vi | 0.087 | 0.462 |
| zh | 0.089 | 0.564 |

**B_pooled** (threshold=0.8776)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.085 | 0.610 |
| de | 0.082 | 0.499 |
| es | 0.074 | 0.537 |
| fr | 0.079 | 0.529 |
| ja | 0.094 | 0.554 |
| vi | 0.087 | 0.462 |
| zh | 0.089 | 0.564 |

**C_per_language** (threshold=0.8689)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.085 | 0.610 |
| de | 0.084 | 0.577 |
| es | 0.079 | 0.609 |
| fr | 0.081 | 0.602 |
| ja | 0.099 | 0.590 |
| vi | 0.090 | 0.579 |
| zh | 0.097 | 0.660 |

**D_shared_slope** (threshold=0.8642)

| lang | realized risk | coverage |
|---|---|---|
| en | 0.085 | 0.610 |
| de | 0.086 | 0.610 |
| es | 0.083 | 0.633 |
| fr | 0.082 | 0.611 |
| ja | 0.101 | 0.628 |
| vi | 0.092 | 0.605 |
| zh | 0.099 | 0.694 |

### Coverage spread across languages (max−min coverage), the headline number

Bootstrap over 2000 resamples of test QUESTIONS (item_idx), percentile 95% CI. This is the number calibration is supposed to shrink: a well-calibrated, language-aware map should push it toward 0.

| calibrator | coverage spread | 95% CI |
|---|---|---|
| raw | 0.148 | [0.111, 0.183] |
| A_english_only | 0.148 | [0.110, 0.183] |
| B_pooled | 0.148 | [0.113, 0.185] |
| C_per_language | 0.083 | [0.058, 0.122] |
| D_shared_slope | 0.089 | [0.070, 0.128] |

### Per-language data cost: calibrator D (shared slope, per-language intercept only) vs. calibrator C (full per-language slope+intercept)

D's shared slope is fixed at the full-training-data value (0.0564, already fit above from all languages pooled); only the intercept is re-estimated per language below, so D needs only 1 free parameter per language vs. C's 2. For each language and sample size n (n train QUESTIONS from that language, all their candidates), we refit C from scratch and refit only D's intercept, then score both on the SAME held-out test split and report ECE, averaged over 5 seeds for n<=100. Full-data column reuses the ECE already reported above.

**de**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1132 | 0.0654 |
| 25 | 0.0944 | 0.0659 |
| 50 | 0.0645 | 0.0571 |
| 100 | 0.0326 | 0.0343 |
| 250 | 0.0181 | 0.0187 |
| all (599) | 0.0240 | 0.0215 |

**es**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1147 | 0.0585 |
| 25 | 0.0717 | 0.0587 |
| 50 | 0.0602 | 0.0631 |
| 100 | 0.0355 | 0.0421 |
| 250 | 0.0287 | 0.0194 |
| all (599) | 0.0210 | 0.0189 |

**fr**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.0969 | 0.0564 |
| 25 | 0.0648 | 0.0562 |
| 50 | 0.0643 | 0.0583 |
| 100 | 0.0386 | 0.0375 |
| 250 | 0.0195 | 0.0178 |
| all (599) | 0.0170 | 0.0162 |

**ja**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1124 | 0.0502 |
| 25 | 0.0790 | 0.0606 |
| 50 | 0.0629 | 0.0578 |
| 100 | 0.0321 | 0.0341 |
| 250 | 0.0294 | 0.0274 |
| all (599) | 0.0202 | 0.0198 |

**vi**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1463 | 0.0495 |
| 25 | 0.0865 | 0.0557 |
| 50 | 0.0542 | 0.0467 |
| 100 | 0.0348 | 0.0262 |
| 250 | 0.0163 | 0.0157 |
| all (599) | 0.0132 | 0.0159 |

**zh**

| n questions | ECE (C) | ECE (D-intercept-only) |
|---|---|---|
| 10 | 0.1020 | 0.0574 |
| 25 | 0.0566 | 0.0521 |
| 50 | 0.0579 | 0.0476 |
| 100 | 0.0318 | 0.0284 |
| 250 | 0.0357 | 0.0298 |
| all (599) | 0.0299 | 0.0298 |

### Parameter cost summary

A: 2 parameters total, fit on English only (7x less per-language data collection than C/D, but does not use or help other languages at all).  
B: 2 parameters total, fit on all 7 languages pooled.  
C: 2 parameters PER language (14 total), each fit ONLY on that language's own labelled data.  
D: 1 shared slope + 7 intercepts (8 total); the slope pools data across all languages and only the intercept needs language-specific labels, and (see learning curve above) a handful of per-language questions is usually enough to estimate one scalar well.


## Verdict

A and B use one function of raw confidence for every language, so they inherit the original problem almost exactly (A/B's per-language coverage numbers match raw's to 3 decimals in both backends): a monotone reparametrization cannot change which candidates rank above a per-language threshold. C (a separate calibrator per language) and D (shared slope, per-language intercept) both reduce the coverage spread; the tables above give the values. They help most for the verifier whose score distribution is well spread (llama-8b) and least for the saturated one (qwen-7b), where most English scores sit within a hair of 1.0 and fitting a logistic map on near-ceiling logits is ill-conditioned. The learning-curve check measures ECE, not coverage: D approaches its full-data ECE with roughly 25-50 labelled per-language questions, while C needs closer to the full budget; note that D's shared slope is fitted on all languages' calibration data, so the per-language count is not its total data requirement. Calibration equalizes scale; it does not guarantee the target-risk threshold transfers, and realized non-English risk under C/D can still run above target for the saturated verifier.