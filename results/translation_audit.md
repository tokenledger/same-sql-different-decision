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
| de | 12.8% | 13.4% | 95 |
| es | 13.2% | 13.5% | 85 |
| fr | 1.6% | 1.7% | 60 |
| ja | 10.1% | 10.5% | 105 |
| vi | 16.9% | 17.0% | 155 |
| zh | 11.7% | 12.0% | 70 |

**big-qwen7b**

| lang | all items | literal-clean only | candidates dropped |
|---|---|---|---|
| de | 4.9% | 5.2% | 95 |
| es | 4.1% | 4.2% | 85 |
| fr | 5.6% | 5.7% | 60 |
| ja | 8.8% | 9.2% | 105 |
| vi | 8.0% | 8.4% | 155 |
| zh | 10.5% | 10.8% | 70 |

Excluding every affected item moves the rates by at most a few tenths of a
point, so the defect is real but does not drive the reported effects. The
scope claim is nevertheless conditional: differences are attributable to the
paired linguistic rendering under MultiSpider's translations, given semantic
equivalence.

