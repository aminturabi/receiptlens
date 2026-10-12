"""ReceiptLens demo API: upload a receipt photo -> structured JSON."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from receiptlens import ReceiptExtractor

app = FastAPI(title="ReceiptLens", version="0.1.0")
extractor: ReceiptExtractor | None = None


@app.on_event("startup")
def _load_model() -> None:
    global extractor
    extractor = ReceiptExtractor()  # downloads weights on first run, then cached


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "donut-base-finetuned-cord-v2"}


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    if extractor is None:
        raise HTTPException(503, "model still loading, try again in a few seconds")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "please upload an image file")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "image too large (max 10 MB)")
    try:
        result = extractor.extract_bytes(data)
    except Exception as exc:  # corrupt image etc.
        raise HTTPException(422, f"could not process image: {exc}")
    return result.model_dump()


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
