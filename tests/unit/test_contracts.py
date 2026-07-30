from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from invoice_ocr.contracts import (
    BoundingBox,
    DetectionRegion,
    InvoiceBatch,
    InvoiceDocument,
    InvoiceHeader,
    InvoiceItem,
    OrientationMetadata,
    Point,
    ProcessingStatus,
)
from invoice_ocr.pipeline import load_invoice_schema, validate_canonical_payload


def test_bbox_requires_positive_extent() -> None:
    with pytest.raises(ValidationError, match="positive width"):
        BoundingBox(x_min=10, y_min=0, x_max=10, y_max=20)


def test_bbox_normalization_clamps_to_layoutlm_range() -> None:
    normalized = BoundingBox(x_min=50, y_min=20, x_max=250, y_max=120).normalize(200, 100)
    assert normalized.model_dump() == {
        "x_min": 250.0,
        "y_min": 200.0,
        "x_max": 1000.0,
        "y_max": 1000.0,
    }


def test_polygon_needs_three_distinct_points() -> None:
    with pytest.raises(ValidationError, match="three distinct"):
        DetectionRegion(
            document_id="document-123",
            source_path="synthetic.png",
            page_index=0,
            model_name="test",
            processing_status=ProcessingStatus.SUCCESS,
            region_id="r0",
            polygon=[Point(x=1, y=1)] * 4,
            bbox=BoundingBox(x_min=1, y_min=1, x_max=2, y_max=2),
            confidence=1,
        )


def test_orientation_metadata_is_explicit() -> None:
    metadata = OrientationMetadata(rotation_degrees=270, confidence=0.9, method="classifier")
    assert metadata.rotation_degrees == 270
    with pytest.raises(ValidationError):
        OrientationMetadata(rotation_degrees=45)


def test_canonical_empty_batch_validates() -> None:
    payload = InvoiceBatch().model_dump(mode="json", exclude_none=False)
    validate_canonical_payload(payload)
    jsonschema.Draft202012Validator.check_schema(load_invoice_schema())


def test_canonical_fixture_validates() -> None:
    fixture = Path("tests/fixtures/synthetic_invoice.json")
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    validate_canonical_payload(payload)


def test_identifiers_remain_strings_and_keep_leading_zeroes() -> None:
    invoice = InvoiceDocument(
        page_number=1,
        invoice=InvoiceHeader(invoice_number="000042"),
        items=[InvoiceItem(line_number=1, lot_number="0007")],
    )
    assert invoice.invoice.invoice_number == "000042"
    assert invoice.items[0].lot_number == "0007"


def test_invoice_count_must_match() -> None:
    with pytest.raises(ValidationError, match="invoice_count"):
        InvoiceBatch(invoice_count=2, invoices=[InvoiceDocument(page_number=1)])
