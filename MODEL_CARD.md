---
license: mit
base_model: naver-clova-ix/donut-base
datasets: [naver-clova-ix/cord-v2]
tags: [donut, document-understanding, receipts, ocr-free]
---
# dociq-donut-cord

Donut (Swin encoder + BART decoder) fine-tuned on CORD v2 to turn a receipt photo into structured JSON
(`menu`, `sub_total`, `total`). Code: https://github.com/Trilokuday3/DocIQ

## Intended use
Extracting line items and totals from receipts similar to CORD (Indonesian retail/restaurant receipts).
Outputs with low confidence or items that do not sum to the subtotal should be reviewed by a human.

## Training
- Base: `naver-clova-ix/donut-base` (not the CORD-finetuned checkpoint)
- Data: CORD v2 train split (800 receipts); model selection on validation (100); test (100) used once
- Image size: _fill_ · max length 768 · lr _fill_ · epochs _fill_ (early stopping on validation TED accuracy) · fp16 on a Kaggle T4

## Results (CORD v2 test, 100 receipts)
| Model | Field F1 | TED acc | JSON valid | Total exact |
|---|---|---|---|---|
| Qwen2.5-VL-3B zero-shot | _fill_ | _fill_ | _fill_ | _fill_ |
| **This model** | _fill_ | _fill_ | _fill_ | _fill_ |
| Donut official CORD checkpoint | _fill_ | _fill_ | _fill_ | _fill_ |

Robustness (F1 drop): blur _fill_ · rotation _fill_ · JPEG q20 _fill_ · 50% brightness _fill_

## Limitations
1,000 training-scale dataset, Indonesian receipts, fixed input resolution; many CORD shop names are blurred for privacy;
may hallucinate fields on unseen layouts. Confidence = exp(mean token log-prob); threshold chosen on validation.
