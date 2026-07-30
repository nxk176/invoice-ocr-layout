"""Money-specific normalization."""

from __future__ import annotations

from decimal import Decimal

from invoice_ocr.postprocessing.numbers import parse_vietnamese_number


def parse_money(value: str | int | float | Decimal | None) -> Decimal | None:
    amount = parse_vietnamese_number(value)
    if amount is None or amount < 0:
        return None
    return amount
