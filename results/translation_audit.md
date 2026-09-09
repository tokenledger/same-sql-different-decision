# Translation fidelity audit

Content that must survive translation for the gold SQL to still answer the
question. `with_english_value` is supposed to hold literal values fixed.

| lang | numeric values preserved | quoted literals preserved | items affected (of 1200) |
|---|---|---|---|
| de | 211/216 = 97.7% | 109/131 = 83.2% | 22 |
| es | 213/216 = 98.6% | 109/131 = 83.2% | 22 |
| fr | 207/216 = 95.8% | 115/131 = 87.8% | 16 |
| ja | 209/216 = 96.8% | 96/131 = 73.3% | 35 |
| vi | 213/216 = 98.6% | 62/131 = 47.3% | 69 |
| zh | 210/216 = 97.2% | 112/131 = 85.5% | 19 |

### Examples where the literal was translated rather than preserved

- **en** Show paragraph details for paragraph with text 'Korea ' .
  **vi** Hiển thị chi tiết về đoạn văn bản có chủ đề là ' Hàn Quốc ' .
- **en** Show names, results and bulgarian commanders of the battles with no ships lost in the 'English Channel'.
  **vi** Hiển thị tên , kết quả và chỉ huy của tất cả các trận chiến ở ' Eo biển Măng-sơ ' và không có mất mát nào về tàu .
- **en** What are the notes of the death events which has substring 'East'?
  **vi** Cho biết những ghi chú có chứa chuỗi con ' Phía Đông ' và liên quan đến các sự kiện thiệt mạng .

### Sensitivity: does the defect drive the results?

Unsafe-promotion rate on held-out databases, all items vs. only items whose
English literal survived translation.

**big-llama8b**

| lang | all items | literal-clean only | candidates dropped |
|---|---|---|---|
| de | 7.4% | 7.7% | 95 |
| es | 9.5% | 9.5% | 85 |
| fr | 0.0% | 0.0% | 60 |
| ja | 7.6% | 7.8% | 105 |
| vi | 12.0% | 12.4% | 155 |
| zh | 9.1% | 9.3% | 70 |

**big-qwen7b**

| lang | all items | literal-clean only | candidates dropped |
|---|---|---|---|
| de | 0.9% | 0.9% | 95 |
| es | 3.0% | 3.1% | 85 |
| fr | 1.8% | 1.8% | 60 |
| ja | 4.7% | 4.9% | 105 |
| vi | 0.4% | 0.4% | 155 |
| zh | 6.2% | 6.3% | 70 |

Excluding every affected item moves the rates by at most a few tenths of a
point, so the defect is real but does not drive the reported effects. The
scope claim is nevertheless conditional: differences are attributable to the
paired linguistic rendering under MultiSpider's translations, given semantic
equivalence.

