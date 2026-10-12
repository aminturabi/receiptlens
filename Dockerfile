# ReceiptLens — Hugging Face Spaces (Docker, CPU)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    no_proxy=localhost,127.0.0.1 \
    NO_PROXY=localhost,127.0.0.1

WORKDIR /code

# torch CPU first (keeps the image lean vs CUDA wheels)
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/

# warm the model into the image so first request is fast
RUN python -c "
import sys; sys.path.insert(0, 'src')
from receiptlens import ReceiptExtractor
ReceiptExtractor()
print('model cached')
"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
