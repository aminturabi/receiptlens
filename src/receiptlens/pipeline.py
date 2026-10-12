"""Core extraction pipeline: image -> Donut (OCR-free) -> normalized JSON."""
from __future__ import annotations

import os
import time
from io import BytesIO

# The sandbox's no_proxy contains bare IPv6 literals that break httpx2's
# proxy-pattern parser inside huggingface_hub. Keep only safe entries.
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import torch
from PIL import Image, ImageEnhance, ImageOps
from transformers import DonutProcessor, VisionEncoderDecoderModel

from .schemas import ExtractionResult, normalize_cord

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"


def preprocess(image: Image.Image) -> Image.Image:
    """Light cleanup for messy scans: orient, grayscale-boost contrast, RGB."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Gentle contrast boost helps thermal-paper / low-ink scans.
    image = ImageEnhance.Contrast(image).enhance(1.15)
    # Donut resizes internally; cap absurd resolutions to bound memory.
    image.thumbnail((1920, 1920), Image.LANCZOS)
    return image


class ReceiptExtractor:
    """Thread-safe-ish single-model extractor (load once, reuse)."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DonutProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self._decoder_input_ids = self.processor.tokenizer(
            TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

    @torch.inference_mode()
    def extract_raw(self, image: Image.Image) -> tuple[dict, float]:
        pixel_values = self.processor(preprocess(image), return_tensors="pt").pixel_values.to(
            self.device
        )
        t0 = time.time()
        out = self.model.generate(
            pixel_values,
            decoder_input_ids=self._decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        latency = time.time() - t0
        seq = self.processor.batch_decode(out.sequences)[0]
        seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )
        return self.processor.token2json(seq), latency

    def extract(self, image: Image.Image) -> ExtractionResult:
        raw, latency = self.extract_raw(image)
        result = normalize_cord(raw if isinstance(raw, dict) else {})
        result.latency_s = round(latency, 2)
        return result

    def extract_bytes(self, data: bytes) -> ExtractionResult:
        return self.extract(Image.open(BytesIO(data)))
