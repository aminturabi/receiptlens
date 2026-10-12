"""ReceiptLens: turn receipt & invoice photos into structured JSON."""

__version__ = "0.1.0"

from .pipeline import ReceiptExtractor  # noqa: F401
from .schemas import ExtractionResult  # noqa: F401
