# What the language shift does to individual execution decisions

One English-calibrated threshold, applied unchanged. Same SQL, same
database, same execution-derived label -- only the question's language
differs, so every flip below is caused by language alone.

## Llama-3.1-8B

Threshold 0.8355 (fit on English, 81 databases). Held-out: 3005 candidates, 2436 correct / 569 incorrect.

| lang | decisions flipped | lost automation | unsafe promotion (of all incorrect) | unsafe promotion (of English-withheld incorrect) | 95% CI, question-clustered |
|---|---|---|---|---|---|
| de | 12.1% | 133/2436 = 5.5% | 42/569 = 7.4% | 9.5% | [4.0%, 11.2%] |
| es | 13.7% | 90/2436 = 3.7% | 54/569 = 9.5% | 12.2% | [5.5%, 13.9%] |
| fr | 13.7% | 332/2436 = 13.6% | 0/569 = 0.0% | 0.0% | 0 of 569 candidate-level opportunities; exact-binomial upper bound 0.6% (unclustered) |
| ja | 16.5% | 236/2436 = 9.7% | 43/569 = 7.6% | 9.7% | [4.3%, 11.2%] |
| vi | 14.4% | 147/2436 = 6.0% | 68/569 = 12.0% | 15.4% | [7.7%, 17.2%] |
| zh | 14.8% | 157/2436 = 6.4% | 52/569 = 9.1% | 11.8% | [5.4%, 13.4%] |

**Transition decomposition.** Counts of held-out candidates by true label and
by (English decision -> target-language decision). `d errors` and `d correct`
are the net change in the accepted set's numerator and denominator, which is
what determines whether aggregate risk moves.

| lang | correct: exec->defer | correct: defer->exec | incorrect: exec->defer | incorrect: defer->exec | d errors | d correct | risk (en) | risk (lang) |
|---|---|---|---|---|---|---|---|---|
| de | 133 | 165 | 24 | 42 | +18 | +32 | 0.094 | 0.103 |
| es | 90 | 255 | 13 | 54 | +41 | +165 | 0.094 | 0.108 |
| fr | 332 | 28 | 51 | 0 | -51 | -304 | 0.094 | 0.076 |
| ja | 236 | 190 | 28 | 43 | +15 | -46 | 0.094 | 0.107 |
| vi | 147 | 186 | 31 | 68 | +37 | +39 | 0.094 | 0.115 |
| zh | 157 | 207 | 28 | 52 | +24 | +50 | 0.094 | 0.106 |

## Qwen2.5-7B

Threshold 1.0000 (fit on English, 81 databases). Held-out: 3005 candidates, 2436 correct / 569 incorrect.

| lang | decisions flipped | lost automation | unsafe promotion (of all incorrect) | unsafe promotion (of English-withheld incorrect) | 95% CI, question-clustered |
|---|---|---|---|---|---|
| de | 22.3% | 583/2436 = 23.9% | 5/569 = 0.9% | 1.0% | [0.0%, 2.6%] |
| es | 18.2% | 423/2436 = 17.4% | 17/569 = 3.0% | 3.5% | [0.6%, 5.9%] |
| fr | 19.1% | 481/2436 = 19.7% | 10/569 = 1.8% | 2.0% | [0.0%, 4.1%] |
| ja | 18.4% | 389/2436 = 16.0% | 27/569 = 4.7% | 5.5% | [2.0%, 8.1%] |
| vi | 22.7% | 593/2436 = 24.3% | 2/569 = 0.4% | 0.4% | [0.0%, 0.9%] |
| zh | 17.5% | 303/2436 = 12.4% | 35/569 = 6.2% | 7.2% | [3.0%, 10.0%] |

**Transition decomposition.** Counts of held-out candidates by true label and
by (English decision -> target-language decision). `d errors` and `d correct`
are the net change in the accepted set's numerator and denominator, which is
what determines whether aggregate risk moves.

| lang | correct: exec->defer | correct: defer->exec | incorrect: exec->defer | incorrect: defer->exec | d errors | d correct | risk (en) | risk (lang) |
|---|---|---|---|---|---|---|---|---|
| de | 583 | 20 | 62 | 5 | -57 | -563 | 0.089 | 0.082 |
| es | 423 | 60 | 46 | 17 | -29 | -363 | 0.089 | 0.100 |
| fr | 481 | 23 | 61 | 10 | -51 | -458 | 0.089 | 0.074 |
| ja | 389 | 92 | 46 | 27 | -19 | -297 | 0.089 | 0.104 |
| vi | 593 | 37 | 50 | 2 | -48 | -556 | 0.089 | 0.106 |
| zh | 303 | 161 | 27 | 35 | +8 | -142 | 0.089 | 0.114 |

## Reading this

`unsafe execution` is the rate at which a wrong query, correctly withheld
in English, is executed automatically once the same request arrives in
another language. `lost automation` is the mirror image: a correct query
needlessly escalated to a human. Both are caused by language alone, since
the SQL and its correctness label are held fixed.

