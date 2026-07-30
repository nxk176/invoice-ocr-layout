"""Business validation that never rewrites OCR values."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from invoice_ocr.contracts import InvoiceDocument, InvoiceValidation


def _close(left: Decimal, right: Decimal, tolerance: Decimal) -> bool:
    return abs(left - right) <= tolerance


def validate_invoice(
    invoice: InvoiceDocument,
    tolerance: Decimal = Decimal("1"),
    required_fields: list[str] | None = None,
) -> InvoiceValidation:
    if tolerance < 0:
        raise ValueError("validation tolerance must be non-negative")
    totals = invoice.totals
    total_check: bool | None = None
    item_check: bool | None = None
    differences: list[Decimal] = []
    if (
        totals.subtotal_excluding_vat is not None
        and totals.vat_total is not None
        and totals.grand_total is not None
    ):
        calculated = totals.subtotal_excluding_vat + totals.vat_total
        total_check = _close(calculated, totals.grand_total, tolerance)
        differences.append(abs(calculated - totals.grand_total))
    amounts = [item.line_amount for item in invoice.items if item.line_amount is not None]
    if totals.subtotal_excluding_vat is not None and amounts:
        calculated_items = sum(amounts, Decimal("0"))
        item_check = _close(calculated_items, totals.subtotal_excluding_vat, tolerance)
        differences.append(abs(calculated_items - totals.subtotal_excluding_vat))
    unresolved = find_unresolved_fields(invoice, required_fields or [])
    result = InvoiceValidation(
        subtotal_plus_vat_equals_grand_total=total_check,
        sum_of_items_equals_subtotal=item_check,
        rounding_difference_detected=any(
            difference > 0 and difference <= tolerance for difference in differences
        ),
        item_count=len(invoice.items),
        unresolved_required_fields=unresolved,
    )
    invoice.validation = result
    return result


def find_unresolved_fields(invoice: InvoiceDocument, required_fields: list[str]) -> list[str]:
    document: Any = invoice.model_dump(mode="python")
    unresolved: list[str] = []
    for dotted_path in required_fields:
        value: Any = document
        for component in dotted_path.split("."):
            if not isinstance(value, dict) or component not in value:
                value = None
                break
            value = value[component]
        if value is None or (isinstance(value, dict) and value.get("needs_review") is True):
            unresolved.append(dotted_path)
    return sorted(set(unresolved))
