# What the language shift does to individual execution decisions

One English-calibrated threshold, applied unchanged. Same SQL, same
database, same execution-derived label; only the question's language
differs, so every flip below is caused by language alone.

## Llama-3.1-8B

Threshold 0.7549 (fit on English, 81 databases). Held-out: 3005 candidates, 2519 correct / 486 incorrect.

| lang | decisions flipped | lost automation | unsafe promotion (of all incorrect) | unsafe promotion (of English-withheld incorrect) | 95% CI, question-clustered |
|---|---|---|---|---|---|
| de | 13.9% | 145/2519 = 5.8% | 62/486 = 12.8% | 17.3% | [7.9%, 17.9%] |
| es | 14.9% | 66/2519 = 2.6% | 64/486 = 13.2% | 17.9% | [8.4%, 18.3%] |
| fr | 14.6% | 352/2519 = 14.0% | 8/486 = 1.6% | 2.2% | [0.2%, 4.3%] |
| ja | 19.5% | 227/2519 = 9.0% | 49/486 = 10.1% | 13.7% | [6.1%, 14.3%] |
| vi | 17.3% | 164/2519 = 6.5% | 82/486 = 16.9% | 22.9% | [11.3%, 23.1%] |
| zh | 15.9% | 179/2519 = 7.1% | 57/486 = 11.7% | 15.9% | [7.4%, 16.9%] |

**Transition decomposition.** Counts of held-out candidates by true label and
by (English decision -> target-language decision). `d errors` and `d correct`
are the net change in the accepted set's numerator and denominator, which is
what determines whether aggregate risk moves.

| lang | correct: exec->defer | correct: defer->exec | incorrect: exec->defer | incorrect: defer->exec | d errors | d correct | risk (en) | risk (lang) |
|---|---|---|---|---|---|---|---|---|
| de | 145 | 199 | 12 | 62 | +50 | +54 | 0.078 | 0.102 |
| es | 66 | 312 | 5 | 64 | +59 | +246 | 0.078 | 0.096 |
| fr | 352 | 29 | 50 | 8 | -42 | -323 | 0.078 | 0.068 |
| ja | 227 | 276 | 33 | 49 | +16 | +49 | 0.078 | 0.085 |
| vi | 164 | 255 | 19 | 82 | +63 | +91 | 0.078 | 0.107 |
| zh | 179 | 213 | 29 | 57 | +28 | +34 | 0.078 | 0.092 |

## Qwen2.5-7B

Threshold 1.0000 (fit on English, 81 databases). Held-out: 3005 candidates, 2519 correct / 486 incorrect.

| lang | decisions flipped | lost automation | unsafe promotion (of all incorrect) | unsafe promotion (of English-withheld incorrect) | 95% CI, question-clustered |
|---|---|---|---|---|---|
| de | 21.7% | 436/2519 = 17.3% | 24/486 = 4.9% | 7.3% | [2.1%, 8.4%] |
| es | 17.4% | 314/2519 = 12.5% | 20/486 = 4.1% | 6.1% | [1.5%, 7.4%] |
| fr | 20.1% | 366/2519 = 14.5% | 27/486 = 5.6% | 8.2% | [2.4%, 9.3%] |
| ja | 21.9% | 371/2519 = 14.7% | 43/486 = 8.8% | 13.0% | [5.2%, 13.0%] |
| vi | 27.2% | 557/2519 = 22.1% | 39/486 = 8.0% | 11.8% | [4.4%, 12.2%] |
| zh | 21.5% | 336/2519 = 13.3% | 51/486 = 10.5% | 15.5% | [6.8%, 14.8%] |

**Transition decomposition.** Counts of held-out candidates by true label and
by (English decision -> target-language decision). `d errors` and `d correct`
are the net change in the accepted set's numerator and denominator, which is
what determines whether aggregate risk moves.

| lang | correct: exec->defer | correct: defer->exec | incorrect: exec->defer | incorrect: defer->exec | d errors | d correct | risk (en) | risk (lang) |
|---|---|---|---|---|---|---|---|---|
| de | 436 | 134 | 57 | 24 | -33 | -302 | 0.085 | 0.082 |
| es | 314 | 131 | 57 | 20 | -37 | -183 | 0.085 | 0.074 |
| fr | 366 | 154 | 58 | 27 | -31 | -212 | 0.085 | 0.079 |
| ja | 371 | 203 | 42 | 43 | +1 | -168 | 0.085 | 0.094 |
| vi | 557 | 148 | 74 | 39 | -35 | -409 | 0.085 | 0.087 |
| zh | 336 | 203 | 57 | 51 | -6 | -133 | 0.085 | 0.089 |

## Reading this

`unsafe execution` is the rate at which a wrong query, correctly withheld
in English, is executed automatically once the same request arrives in
another language. `lost automation` is the mirror image: a correct query
needlessly escalated to a human. Both are caused by language alone, since
the SQL and its correctness label are held fixed.

