"""ReceiptLens on Modal — fast GPU inference, free tier.

Deploy: modal deploy modal_app.py  (needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)
"""
import io
import json
import modal

app = modal.App("receiptlens")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1",
    "transformers==4.48.0",
    "Pillow==11.1.0",
    "huggingface_hub==0.27.1",
)

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    scaledown_window=300,  # keep warm 5 min after last request
)
@modal.web_endpoint(method="POST")
def extract(request: dict):
    """Accept {'image_b64': '<base64>'} -> {'items': [...], 'subtotal':, 'tax':, 'total':, 'raw': {...}}"""
    import base64
    import re
    import torch
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    # lazy global model (loaded once per container)
    global _model, _processor
    if "_model" not in globals():
        _processor = DonutProcessor.from_pretrained(MODEL_ID)
        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to("cuda").eval()

    img = Image.open(io.BytesIO(base64.b64decode(request["image_b64"]))).convert("RGB")
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = _processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    pixel_values = _processor(img, return_tensors="pt").pixel_values.to("cuda")

    with torch.no_grad():
        outputs = _model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=384,
            early_stopping=True,
            pad_token_id=_processor.tokenizer.pad_token_id,
            eos_token_id=_processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    seq = _processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(_processor.tokenizer.eos_token, "").replace(
        _processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()
    raw = _processor.token2json(seq)

    # normalize (mirrors src/receiptlens/schemas.py)
    def first(*vs):
        for v in vs:
            s = str(v or "").strip()
            if s:
                return s
        return None

    items = []

    def collect(node):
        arr = node if isinstance(node, list) else ([node] if node else [])
        for m in arr:
            if not isinstance(m, dict):
                continue
            name = first(m.get("nm"), m.get("name")) or ""
            price = first(m.get("price"))
            if name and not re.search(r"tel|street|road|ave\b|blvd|lane|thank you", name, re.I):
                items.append({
                    "name": name,
                    "quantity": first(m.get("cnt"), m.get("quantity")),
                    "unit_price": first(m.get("unitprice"), m.get("unit_price")),
                    "price": price,
                })
            if m.get("sub"):
                collect(m["sub"])

    collect(raw.get("menu"))
    sub = raw.get("subtotal") or raw.get("sub_total") or {}
    tot = raw.get("total") or {}
    return {
        "items": items,
        "subtotal": first(sub.get("subtotal_price")),
        "tax": first(sub.get("tax_price")),
        "total": first(tot.get("total_price")),
        "raw": raw,
    }
