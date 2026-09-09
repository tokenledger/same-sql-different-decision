# Do the mitigations restore decisions, or only coverage?

Decision-level metrics for the bilingual and pivot conditions, against the
English reference policy at the English-calibrated threshold. Held-out
databases only. `flip` counts identical SQL whose execute/defer decision
differs from English; `overlap` is the Jaccard similarity of the accepted
sets.

## Llama-3.1-8B

Threshold 0.8355. Held-out 3005 candidates (2436 correct / 569 incorrect); English coverage 0.450, risk 0.094.

| condition | flip | unsafe promoted | lost automation | overlap | coverage | risk |
|---|---|---|---|---|---|---|
| native | 14.2% | 7.6% | 7.5% | 0.727 | 0.451 | 0.103 |
| bilingual | 14.0% | 3.5% | 10.9% | 0.712 | 0.383 | 0.087 |
| pivot | 10.5% | 1.6% | 8.6% | 0.780 | 0.395 | 0.089 |

Per language, flip rate:

| condition | de | es | fr | ja | vi | zh |
|---|---|---|---|---|---|---|
| native | 12.1% | 13.7% | 13.7% | 16.5% | 14.4% | 14.8% |
| bilingual | 14.2% | 13.2% | 16.3% | 14.2% | 13.3% | 12.4% |
| pivot | 8.7% | 8.6% | 9.0% | 11.6% | 13.8% | 11.1% |

## Qwen2.5-7B

Threshold 1.0000. Held-out 3005 candidates (2436 correct / 569 incorrect); English coverage 0.304, risk 0.089.

| condition | flip | unsafe promoted | lost automation | overlap | coverage | risk |
|---|---|---|---|---|---|---|
| native | 19.7% | 2.8% | 19.0% | 0.400 | 0.161 | 0.097 |
| bilingual | 15.0% | 7.6% | 4.9% | 0.635 | 0.369 | 0.103 |
| pivot | 11.9% | 1.4% | 10.1% | 0.643 | 0.241 | 0.085 |

Per language, flip rate:

| condition | de | es | fr | ja | vi | zh |
|---|---|---|---|---|---|---|
| native | 22.3% | 18.2% | 19.1% | 18.4% | 22.7% | 17.5% |
| bilingual | 16.1% | 13.6% | 13.6% | 16.6% | 14.0% | 16.4% |
| pivot | 11.6% | 10.1% | 10.0% | 12.2% | 15.5% | 11.8% |

A mitigation that lowers coverage spread but leaves `flip` and `overlap`
unchanged has equalised how many queries run without restoring which ones.

