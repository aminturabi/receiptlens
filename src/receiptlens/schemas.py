"""Clean output schema for ReceiptLens extractions."""
from __future__ import annotations

import re
from typing import Any, Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    name: str = ""
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    price: Optional[str] = None


class ExtractionResult(BaseModel):
    """Normalized, frontend-friendly extraction result."""

    merchant: Optional[str] = None
    date: Optional[str] = None
    items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[str] = None
    tax: Optional[str] = None
    total: Optional[str] = None
    currency: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict, description="Raw CORD-v2 JSON from the model")
    latency_s: float = 0.0


def _first(*vals: Any) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


_ADDRESS_RE = re.compile(r"tel|street|road|ave\b|blvd|lane|thank you", re.IGNORECASE)
_PHONE_RE = re.compile(r"^\d{3}-\d{4}$")


def _looks_like_address(name: str, price: str | None) -> bool:
    return bool(_ADDRESS_RE.search(name or "")) or bool(price and _PHONE_RE.match(price))


def _collect_items(node: Any, items: list[LineItem]) -> None:
    """Recursively collect menu items — CORD-v2 nests them under `sub`."""
    if isinstance(node, dict):
        node = [node]
    if not isinstance(node, list):
        return
    for m in node:
        if not isinstance(m, dict):
            continue
        name = _first(m.get("nm"), m.get("name")) or ""
        price = _first(m.get("price"))
        if name and not _looks_like_address(name, price):
            items.append(
                LineItem(
                    name=name,
                    quantity=_first(m.get("cnt"), m.get("quantity")),
                    unit_price=_first(m.get("unitprice"), m.get("unit_price")),
                    price=price,
                )
            )
        # nested sub-items (CORD-v2 groups them under `sub`)
        sub = m.get("sub")
        if sub:
            _collect_items(sub, items)


def normalize_cord(raw: dict[str, Any]) -> ExtractionResult:
    """Map the model's raw CORD-v2 JSON onto the clean schema."""
    items: list[LineItem] = []
    _collect_items(raw.get("menu"), items)

    sub = raw.get("subtotal") or raw.get("sub_total") or {}
    tot = raw.get("total") or {}

    return ExtractionResult(
        merchant=_first((raw.get("store") or {}).get("name") if isinstance(raw.get("store"), dict) else None),
        date=_first((raw.get("store") or {}).get("date") if isinstance(raw.get("store"), dict) else None),
        items=items,
        subtotal=_first(sub.get("subtotal_price")) if isinstance(sub, dict) else None,
        tax=_first(sub.get("tax_price")) if isinstance(sub, dict) else None,
        total=_first(tot.get("total_price")) if isinstance(tot, dict) else None,
        currency=_first(tot.get("currency")) if isinstance(tot, dict) else None,
        raw=raw,
    )
