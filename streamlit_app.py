"""ReceiptLens on Streamlit Cloud — server-side ONNX inference.

Free tier: 1GB RAM. Uses the lightweight quantized ONNX model (~200MB).
"""
import io
import time

import streamlit as st
from PIL import Image

st.set_page_config(page_title="ReceiptLens", page_icon="🧾", layout="wide")

MODEL_ID = "Xenova/donut-base-finetuned-cord-v2"


@st.cache_resource(show_spinner="Loading AI model (one time)...")
def load_model():
    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = ORTModelForVision2Seq.from_pretrained(MODEL_ID)
    return processor, tokenizer, model


def token2json_simple(seq):
    """Minimal Donut tag parser -> dict (kept small for the demo)."""
    import re

    seq = re.sub(r"^<s_cord-v2>", "", seq).strip()

    def parse(s, pos):
        m = re.match(r"<s_([^>]+)>", s[pos:])
        tag = m.group(1)
        pos += len(m.group(0))
        children, texts, groups, cur = {}, [], [], None
        while pos < len(s):
            if s.startswith("</s_", pos):
                c = re.match(r"</s_([^>]+)>", s[pos:])
                pos += len(c.group(0))
                if c.group(1) == tag:
                    break
                continue
            if s.startswith("<sep/>", pos):
                pos += 6
                if cur:
                    groups.append(cur)
                cur = {}
                continue
            cm = re.match(r"<s_([^>]+)>", s[pos:])
            if cm:
                val, pos = parse(s, pos)
                cur = cur or {}
                if cm.group(1) in cur:
                    if not isinstance(cur[cm.group(1)], list):
                        cur[cm.group(1)] = [cur[cm.group(1)]]
                    cur[cm.group(1)].append(val)
                else:
                    cur[cm.group(1)] = val
            else:
                texts.append(s[pos])
                pos += 1
        if cur:
            groups.append(cur)
        if len(groups) > 1:
            return groups, pos
        if groups:
            return groups[0], pos
        if cur:
            return cur, pos
        return "".join(texts).strip(), pos

    val, _ = parse("<s_root>" + seq + "</s_root>", 0)
    return val


def normalize(raw):
    items = []

    def collect(node):
        arr = node if isinstance(node, list) else ([node] if node else [])
        for m in arr:
            if not isinstance(m, dict):
                continue
            name = (m.get("nm") or m.get("name") or "").strip()
            price = (m.get("price") or "").strip()
            if name and not __import__("re").search(
                r"tel|street|road|ave\b|blvd|lane|thank you", name, __import__("re").I
            ):
                items.append({
                    "name": name,
                    "quantity": (m.get("cnt") or m.get("quantity") or "").strip() or None,
                    "unit_price": (m.get("unitprice") or m.get("unit_price") or "").strip() or None,
                    "price": price or None,
                })
            if m.get("sub"):
                collect(m["sub"])

    collect(raw.get("menu"))
    sub = raw.get("subtotal") or raw.get("sub_total") or {}
    tot = raw.get("total") or {}
    return {
        "items": items,
        "subtotal": (sub.get("subtotal_price") or "").strip() or None,
        "tax": (sub.get("tax_price") or "").strip() or None,
        "total": (tot.get("total_price") or "").strip() or None,
        "raw": raw,
    }


st.title("ReceiptLens 🧾 — receipts → JSON")
st.write(
    "Upload a receipt photo. The Donut vision transformer reads it and returns "
    "structured data: line items, subtotal, tax, total."
)

uploaded = st.file_uploader("Choose a receipt image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded receipt", width=350)

    if st.button("Extract structured data", type="primary"):
        processor, tokenizer, model = load_model()
        with st.spinner("AI is reading your receipt..."):
            t0 = time.time()
            pixel_values = processor(img, return_tensors="pt").pixel_values
            decoder_input_ids = tokenizer(
                "<s_cord-v2>", add_special_tokens=False, return_tensors="pt"
            ).input_ids
            output = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=384,
            )
            dt = time.time() - t0
            seq = tokenizer.batch_decode(output)[0]
            seq = seq.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
            result = normalize(token2json_simple(seq))

        st.success(f"Done in {dt:.1f}s")
        st.subheader("Line items")
        st.table([{"Item": i["name"], "Price": i["price"] or "–"}
                  for i in result["items"]])
        c1, c2, c3 = st.columns(3)
        c1.metric("Subtotal", result["subtotal"] or "–")
        c2.metric("Tax", result["tax"] or "–")
        c3.metric("Total", result["total"] or "–")

        st.subheader("Raw JSON")
        st.json(result["raw"])

        # CSV download
        import csv

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Item", "Quantity", "Unit Price", "Price"])
        for i in result["items"]:
            w.writerow([i["name"], i["quantity"], i["unit_price"], i["price"]])
        w.writerow([])
        w.writerow(["Subtotal", "", "", result["subtotal"]])
        w.writerow(["Tax", "", "", result["tax"]])
        w.writerow(["Total", "", "", result["total"]])
        st.download_button("Download CSV", buf.getvalue(), "receipt.csv", "text/csv")
else:
    st.info("👆 Upload a receipt to begin.")
