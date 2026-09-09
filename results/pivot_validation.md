# Is the pivot improvement genuine, or benchmark repair?

`clean` restricts to items whose English literal survived MultiSpider's own
translation, so machine translation has no defect to repair there. If pivot's
improvement holds on `clean`, it is not an artifact of fixing the benchmark.

## Llama-3.1-8B

Held-out 3005 candidates (2519 correct / 486 incorrect), threshold 0.7549.

| lang | flip native | flip pivot | Δ flip [95% CI] | unsafe native | unsafe pivot | Δ flip, literal-clean | n dropped |
|---|---|---|---|---|---|---|---|
| de | 13.9% | 9.8% | -4.1% [-6.3%, -1.9%] * | 12.8% | 2.7% | -4.0% | 95 |
| es | 14.9% | 9.9% | -5.0% [-7.4%, -2.5%] * | 13.2% | 1.9% | -4.9% | 85 |
| fr | 14.6% | 10.1% | -4.5% [-7.0%, -2.0%] * | 1.6% | 2.3% | -4.6% | 60 |
| ja | 19.5% | 13.2% | -6.3% [-9.3%, -3.3%] * | 10.1% | 2.3% | -6.0% | 105 |
| vi | 17.3% | 17.2% | -0.1% [-3.2%, +3.1%] | 16.9% | 3.7% | -1.4% | 155 |
| zh | 15.9% | 12.6% | -3.3% [-5.7%, -0.7%] * | 11.7% | 3.7% | -3.1% | 70 |

- Languages improved: **6/6**
- Worst per-language flip: 19.5% → **17.2%**
- Worst per-language unsafe promotion: 16.9% → **3.7%**

## Qwen2.5-7B

Held-out 3005 candidates (2519 correct / 486 incorrect), threshold 1.0000.

| lang | flip native | flip pivot | Δ flip [95% CI] | unsafe native | unsafe pivot | Δ flip, literal-clean | n dropped |
|---|---|---|---|---|---|---|---|
| de | 21.7% | 15.3% | -6.4% [-9.6%, -3.1%] * | 4.9% | 2.7% | -6.5% | 95 |
| es | 17.4% | 14.0% | -3.4% [-6.3%, -0.5%] * | 4.1% | 3.9% | -3.4% | 85 |
| fr | 20.1% | 14.9% | -5.2% [-8.5%, -1.8%] * | 5.6% | 3.3% | -5.1% | 60 |
| ja | 21.9% | 20.1% | -1.8% [-4.8%, +1.5%] | 8.8% | 2.1% | -2.2% | 105 |
| vi | 27.2% | 23.1% | -4.2% [-7.7%, -0.6%] * | 8.0% | 2.7% | -3.9% | 155 |
| zh | 21.5% | 18.2% | -3.3% [-6.4%, -0.4%] * | 10.5% | 4.1% | -3.4% | 70 |

- Languages improved: **6/6**
- Worst per-language flip: 27.2% → **23.1%**
- Worst per-language unsafe promotion: 10.5% → **4.1%**

A `Δ flip, literal-clean` close to the all-items `Δ flip` means the
improvement is not explained by machine translation repairing values that
MultiSpider's own translation had altered.

