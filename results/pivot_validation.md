# Is the pivot improvement genuine, or benchmark repair?

`clean` restricts to items whose English literal survived MultiSpider's own
translation, so machine translation has no defect to repair there. If pivot's
improvement holds on `clean`, it is not an artifact of fixing the benchmark.

## Llama-3.1-8B

Held-out 3005 candidates (2436 correct / 569 incorrect), threshold 0.8355.

| lang | flip native | flip pivot | Δ flip [95% CI] | unsafe native | unsafe pivot | Δ flip, literal-clean | n dropped |
|---|---|---|---|---|---|---|---|
| de | 12.1% | 8.7% | -3.4% [-5.5%, -1.5%] * | 7.4% | 1.6% | -3.1% | 95 |
| es | 13.7% | 8.6% | -5.1% [-7.8%, -2.3%] * | 9.5% | 1.1% | -5.1% | 85 |
| fr | 13.7% | 9.0% | -4.7% [-7.8%, -1.8%] * | 0.0% | 2.1% | -4.6% | 60 |
| ja | 16.5% | 11.6% | -5.0% [-7.8%, -2.2%] * | 7.6% | 1.8% | -4.5% | 105 |
| vi | 14.4% | 13.8% | -0.6% [-3.4%, +2.8%] | 12.0% | 1.2% | -1.4% | 155 |
| zh | 14.8% | 11.1% | -3.7% [-6.7%, -0.7%] * | 9.1% | 2.1% | -3.6% | 70 |

- Languages improved: **6/6**
- Worst per-language flip: 16.5% → **13.8%**
- Worst per-language unsafe promotion: 12.0% → **2.1%**

## Qwen2.5-7B

Held-out 3005 candidates (2436 correct / 569 incorrect), threshold 1.0000.

| lang | flip native | flip pivot | Δ flip [95% CI] | unsafe native | unsafe pivot | Δ flip, literal-clean | n dropped |
|---|---|---|---|---|---|---|---|
| de | 22.3% | 11.6% | -10.6% [-13.6%, -8.0%] * | 0.9% | 0.7% | -11.0% | 95 |
| es | 18.2% | 10.1% | -8.1% [-10.9%, -5.2%] * | 3.0% | 0.4% | -8.3% | 85 |
| fr | 19.1% | 10.0% | -9.2% [-12.3%, -6.1%] * | 1.8% | 0.9% | -9.3% | 60 |
| ja | 18.4% | 12.2% | -6.3% [-8.3%, -4.2%] * | 4.7% | 2.1% | -6.4% | 105 |
| vi | 22.7% | 15.5% | -7.2% [-10.0%, -4.3%] * | 0.4% | 1.6% | -7.6% | 155 |
| zh | 17.5% | 11.8% | -5.7% [-8.9%, -2.4%] * | 6.2% | 2.8% | -5.8% | 70 |

- Languages improved: **6/6**
- Worst per-language flip: 22.7% → **15.5%**
- Worst per-language unsafe promotion: 6.2% → **2.8%**

A `Δ flip, literal-clean` close to the all-items `Δ flip` means the
improvement is not explained by machine translation repairing values that
MultiSpider's own translation had altered.

