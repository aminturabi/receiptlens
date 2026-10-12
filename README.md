# ReceiptLens 🧾 → JSON

Turn receipt & invoice **photos into structured JSON** — line items, subtotal, tax, total — with one API call. No OCR pipeline, no templates, no manual rules.

**Live demo:** _(link after deploy)_ · **API:** `POST /api/extract`

## How it works

```
receipt photo
    │  preprocess (orient, contrast boost, resize cap)
    ▼
Donut — OCR-free document understanding transformer
  (Swin encoder + BART decoder, fine-tuned on CORD receipts)
    │  task prompt <s_cord-v2> → structured token sequence
    ▼
normalize → clean JSON schema {merchant, items[], subtotal, tax, total}
```

Classical receipt pipelines chain *detect → OCR → serialize → parse*, and every stage
adds errors. Donut reads the image end-to-end and emits the structure directly,
which is why it stays robust on messy scans (rotation, blur, thermal-paper fade).

## Evaluation — field-level, not just "accuracy"

Scored on a held-out set of **24 receipts** (16 clean + 8 degraded: rotation, blur,
brightness jitter) with exact ground truth:

| Field | Metric | Score |
|---|---|---|
| Item name | F1 (P / R) | **0.62** (0.76 / 0.53) |
| Item price (on matched items) | Accuracy | 0.29 |
| Subtotal | Accuracy | 0.67 |
| Tax | Accuracy | 0.71 |
| Total | Accuracy | **0.92** |
| Latency | mean / receipt (CPU) | 19 s |

Reproduce: `python eval/make_synth.py && python eval/evaluate.py eval/data/synth`

## Run it locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port 7860
# open http://localhost:7860
```

```python
from receiptlens import ReceiptExtractor
ex = ReceiptExtractor()          # downloads weights once, then cached
result = ex.extract_bytes(open("receipt.jpg","rb").read())
print(result.model_dump_json(indent=2))
```

## API

`POST /api/extract` — multipart form with an `image` file (≤10 MB) →

```json
{
  "merchant": "FRESH MART",
  "items": [{"name": "Milk 1L", "quantity": null, "unit_price": null, "price": "3.50"}],
  "subtotal": "15.55", "tax": "0.78", "total": "16.33",
  "latency_s": 6.4
}
```

`GET /health` → `{"status": "ok"}`

## Project structure

```
src/receiptlens/   pipeline.py (model + preprocess), schemas.py (clean JSON schema)
app/               FastAPI + single-page demo frontend
eval/              make_synth.py (eval-set generator), evaluate.py (field-level F1)
Dockerfile         Hugging Face Spaces deploy
```

## Credits

- Model: [naver-clova-ix/donut-base-finetuned-cord-v2](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2) (MIT)
- Paper: *OCR-free Document Understanding Transformer* (Kim et al., ECCV 2022)

Built by [Amin Khan](https://aminkhanofficial.vercel.app/).
