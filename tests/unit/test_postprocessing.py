from __future__ import annotations

from decimal import Decimal

import pytest

from invoice_ocr.contracts import (
    BoundingBox,
    InvoiceDocument,
    InvoiceItem,
    InvoiceTotals,
    LabeledEntity,
    ProcessingStatus,
    TableCell,
)
from invoice_ocr.postprocessing.dates import normalize_date
from invoice_ocr.postprocessing.fields import entities_to_invoice
from invoice_ocr.postprocessing.medicine_description import parse_medicine_description
from invoice_ocr.postprocessing.numbers import parse_vietnamese_number
from invoice_ocr.postprocessing.validation import validate_invoice
from invoice_ocr.reconstruction.medicine_rows import reconstruct_medicine_item, reconstruct_rows


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("09/09/2025", "2025-09-09"),
        ("2025-09-09", "2025-09-09"),
        ("ngày 9 tháng 9 năm 2025", "2025-09-09"),
        ("31/02/2025", None),
        ("09/09/25", None),
        (None, None),
    ],
)
def test_date_normalization(raw: str | None, expected: str | None) -> None:
    assert normalize_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.234.567", Decimal("1234567")),
        ("1.234,50", Decimal("1234.50")),
        ("1,234.50", Decimal("1234.50")),
        ("12,5", Decimal("12.5")),
        ("1 234 567", Decimal("1234567")),
        ("không rõ", None),
    ],
)
def test_vietnamese_number_parsing(raw: str, expected: Decimal | None) -> None:
    assert parse_vietnamese_number(raw) == expected


def test_nsx_country_does_not_guess_manufacturer() -> None:
    parsed = parse_medicine_description("Thuốc Tổng Hợp 500mg; NSX: Việt Nam")
    assert parsed["medicine_name"] == "Thuốc Tổng Hợp"
    assert parsed["strength"] == "500mg"
    assert parsed["country_of_manufacture"] == "Việt Nam"
    assert parsed["manufacturer"] is None


def make_entity(label: str, text: str) -> LabeledEntity:
    return LabeledEntity(
        document_id="document-123",
        source_path="synthetic.png",
        page_index=0,
        model_name="mock-layout",
        model_revision="test",
        processing_status=ProcessingStatus.SUCCESS,
        entity_id=f"entity-{label}",
        label=label,
        text=text,
        bbox=BoundingBox(x_min=1, y_min=1, x_max=20, y_max=10),
        confidence=1,
    )


def test_null_no_guess_and_workflow_default_provenance() -> None:
    invoice = entities_to_invoice(
        1,
        [make_entity("SUPPLIER_NAME", "Synthetic Supplier")],
        {"status": "NEW", "invoice_type": None},
    )
    assert invoice.invoice.invoice_number is None
    assert invoice.workflow_fields.status.value == "NEW"
    assert invoice.workflow_fields.status.source == "workflow_default"
    assert invoice.workflow_fields.delivery_unit.value is None
    assert invoice.workflow_fields.delivery_unit.suggested_value is None


def test_validation_arithmetic_does_not_rewrite_values() -> None:
    invoice = InvoiceDocument(
        page_number=1,
        items=[
            InvoiceItem(line_number=1, line_amount=Decimal("60")),
            InvoiceItem(line_number=2, line_amount=Decimal("40")),
        ],
        totals=InvoiceTotals(
            subtotal_excluding_vat=Decimal("100"),
            vat_total=Decimal("10"),
            grand_total=Decimal("110"),
        ),
    )
    result = validate_invoice(invoice, Decimal("0.01"), ["invoice.invoice_number"])
    assert result.subtotal_plus_vat_equals_grand_total is True
    assert result.sum_of_items_equals_subtotal is True
    assert result.unresolved_required_fields == ["invoice.invoice_number"]
    assert invoice.totals.grand_total == Decimal("110")


def test_table_row_reconstruction_preserves_lot_string() -> None:
    base = {
        "document_id": "document-123",
        "source_path": "synthetic.png",
        "page_index": 0,
        "model_name": "mock",
        "model_revision": "test",
        "processing_status": ProcessingStatus.SUCCESS,
        "table_id": "table-1",
        "row_index": 0,
        "bbox": BoundingBox(x_min=1, y_min=1, x_max=2, y_max=2),
    }
    cells = [
        TableCell(**base, column_index=0, text="Thuốc Tổng Hợp", label="RAW_DESCRIPTION"),
        TableCell(**base, column_index=1, text="0007", label="LOT_NUMBER"),
        TableCell(**base, column_index=2, text="1.234", label="LINE_AMOUNT"),
    ]
    row = reconstruct_rows(cells)[0]
    item = reconstruct_medicine_item(row, 1)
    assert item.raw_description == "Thuốc Tổng Hợp"
    assert item.lot_number == "0007"
    assert item.line_amount == Decimal("1234")
