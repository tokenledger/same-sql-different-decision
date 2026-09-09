# Do the mitigations restore decisions, or only coverage?

Decision-level metrics for the bilingual and pivot conditions, against the
English reference policy at the English-calibrated threshold. Held-out
databases only. `flip` counts identical SQL whose execute/defer decision
differs from English; `overlap` is the Jaccard similarity of the accepted
sets.

## Llama-3.1-8B

Threshold 0.7549. Held-out 3005 candidates (2519 correct / 486 incorrect); English coverage 0.544, risk 0.078.

| condition | flip | unsafe promoted | lost automation | overlap | coverage | risk |
|---|---|---|---|---|---|---|
| native | 16.0% | 11.0% | 7.5% | 0.747 | 0.562 | 0.092 |
| bilingual | 14.5% | 6.6% | 9.4% | 0.755 | 0.499 | 0.073 |
| pivot | 12.2% | 2.7% | 9.0% | 0.791 | 0.495 | 0.075 |

Per language, flip rate:

| condition | de | es | fr | ja | vi | zh |
|---|---|---|---|---|---|---|
| native | 13.9% | 14.9% | 14.6% | 19.5% | 17.3% | 15.9% |
| bilingual | 15.1% | 12.6% | 15.7% | 15.5% | 14.5% | 13.5% |
| pivot | 9.8% | 9.9% | 10.1% | 13.2% | 17.2% | 12.6% |

## Qwen2.5-7B

Threshold 1.0000. Held-out 3005 candidates (2519 correct / 486 incorrect); English coverage 0.610, risk 0.085.

| condition | flip | unsafe promoted | lost automation | overlap | coverage | risk |
|---|---|---|---|---|---|---|
| native | 21.6% | 7.0% | 15.7% | 0.680 | 0.524 | 0.084 |
| bilingual | 16.9% | 10.5% | 5.9% | 0.765 | 0.657 | 0.087 |
| pivot | 17.6% | 3.1% | 13.8% | 0.729 | 0.513 | 0.071 |

Per language, flip rate:

| condition | de | es | fr | ja | vi | zh |
|---|---|---|---|---|---|---|
| native | 21.7% | 17.4% | 20.1% | 21.9% | 27.2% | 21.5% |
| bilingual | 17.9% | 14.4% | 16.0% | 17.5% | 18.4% | 17.0% |
| pivot | 15.3% | 14.0% | 14.9% | 20.1% | 23.1% | 18.2% |

A mitigation that lowers coverage spread but leaves `flip` and `overlap`
unchanged has equalised how many queries run without restoring which ones.

