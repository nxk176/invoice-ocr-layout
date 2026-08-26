from __future__ import annotations

from invoice_ocr.contracts import BoundingBox
from invoice_ocr.layout_gt.alignment import (
    GroundTruthField,
    OCRRegion,
    align_ground_truth_fields,
    flatten_canonical_ground_truth,
    normalize_layout_bbox,
)


def _region(
    region_id: str,
    text: str,
    x: float,
    y: float,
    page_index: int = 0,
) -> OCRRegion:
    box = BoundingBox(x_min=x, y_min=y, x_max=x + 40, y_max=y + 10)
    return OCRRegion(
        region_id=region_id,
        page_index=page_index,
        text=text,
        bbox=box,
        polygon=[
            [box.x_min, box.y_min],
            [box.x_max, box.y_min],
            [box.x_max, box.y_max],
            [box.x_min, box.y_max],
        ],
        detection_confidence=0.9,
        recognition_confidence=0.8,
    )


def _field(
    path: str,
    label: str,
    value: str,
    *,
    kind: str = "text",
    page_index: int = 0,
    item_index: int | None = None,
) -> GroundTruthField:
    return GroundTruthField(
        field_path=path,
        label=label,
        gt_value=value,
        normalization_kind=kind,
        page_index=page_index,
        invoice_index=0,
        item_index=item_index,
        line_number=item_index + 1 if item_index is not None else None,
    )


def test_flatten_canonical_gt_covers_invoice_workflow_parties_items_and_totals() -> None:
    payload = {
        "document_type": "VAT_INVOICE_BATCH",
        "invoice_count": 1,
        "invoices": [
            {
                "page_number": 1,
                "workflow_fields": {
                    "status": {"value": "Printed status", "source": "invoice"},
                    "invoice_type": {
                        "value": "Default type",
                        "source": "workflow_default",
                    },
                    "bid_package": {
                        "value": "Synthetic package",
                        "source": "invoice",
                    },
                },
                "invoice": {
                    "invoice_number": "SYN-001",
                    "invoice_date": "2026-01-02",
                },
                "supplier": {"supplier_name": "Synthetic Supplier", "tax_code": "0001"},
                "buyer": {"buyer_organization": "Synthetic Buyer"},
                "items": [
                    {
                        "line_number": 1,
                        "medicine_name": "Synthetic Medicine",
                        "quantity": 2,
                        "line_amount": 2000,
                    }
                ],
                "totals": {"grand_total": 2000},
                "validation": {"item_count": 1},
            }
        ],
        "_review": {"synthetic": True},
    }
    fields = flatten_canonical_ground_truth(payload)
    labels = {field.label for field in fields}
    assert {
        "STATUS",
        "BID_PACKAGE",
        "INVOICE_NUMBER",
        "INVOICE_DATE",
        "SUPPLIER_NAME",
        "SELLER_TAX_CODE",
        "BUYER_ORGANIZATION",
        "LINE_NUMBER",
        "MEDICINE_NAME",
        "QUANTITY",
        "LINE_AMOUNT",
        "GRAND_TOTAL",
    } <= labels
    assert "INVOICE_TYPE" not in labels
    assert all("validation" not in field.field_path for field in fields)


def test_exact_normalized_multibox_and_page_aware_matching() -> None:
    fields = [
        _field("number", "INVOICE_NUMBER", "SYN-001", kind="identifier", page_index=1),
        _field("supplier", "SUPPLIER_NAME", "Synthetic Supplier"),
        _field("amount", "GRAND_TOTAL", "1234", kind="money"),
        _field("date", "INVOICE_DATE", "2026-01-02", kind="date"),
    ]
    regions = [
        _region("wrong-page", "SYN-001", 10, 10, page_index=0),
        _region("right-page", "SYN-001", 10, 10, page_index=1),
        _region("supplier-a", "Synthetic", 10, 30),
        _region("supplier-b", "Supplier", 55, 30),
        _region("amount", "1.234", 100, 80),
        _region("date", "02/01/2026", 100, 100),
    ]
    result = align_ground_truth_fields(fields, regions)
    by_label = {match.field.label: match for match in result.matches}
    assert by_label["INVOICE_NUMBER"].candidate.regions[0].region_id == "right-page"
    assert by_label["SUPPLIER_NAME"].match_method == "multi_box_exact"
    assert result.region_labels["supplier-a"] == "B-SUPPLIER_NAME"
    assert result.region_labels["supplier-b"] == "I-SUPPLIER_NAME"
    assert "normalized" in by_label["GRAND_TOTAL"].match_method
    assert by_label["INVOICE_DATE"].match_method == "date_normalized"


def test_known_field_prefix_produces_unique_training_eligible_match() -> None:
    field = _field(
        "number",
        "INVOICE_NUMBER",
        "00000123",
        kind="identifier",
    )
    result = align_ground_truth_fields(
        [field],
        [_region("number", "S\u1ed1: 00000123", 10, 10)],
    )

    match = result.matches[0]
    assert match.match_method == "anchored_exact_normalized"
    assert match.candidate_count == 1
    assert match.training_eligible is True
    assert result.region_labels == {"number": "B-INVOICE_NUMBER"}


def test_duplicate_and_fuzzy_candidates_are_ambiguous_and_not_training_labels() -> None:
    duplicate = _field("number", "INVOICE_NUMBER", "SYN-009", kind="identifier")
    fuzzy = _field("supplier", "SUPPLIER_NAME", "Synthetic Supplier")
    regions = [
        _region("number-a", "SYN-009", 10, 10),
        _region("number-b", "SYN-009", 10, 50),
        _region("supplier", "Synthetic Suplier", 10, 80),
    ]
    result = align_ground_truth_fields([duplicate, fuzzy], regions)
    matches = {match.field.field_path: match for match in result.matches}
    assert matches["number"].ambiguous is True
    assert matches["number"].duplicate_candidates is True
    assert matches["supplier"].ambiguous is True
    assert result.region_labels == {}


def test_number_matching_rejects_value_embedded_in_non_numeric_text() -> None:
    field = _field(
        "vat",
        "ITEM_VAT_RATE",
        "5",
        kind="number",
        item_index=0,
    )
    description = _region("description", "Synthetic solution 5ml", 10, 10)

    rejected = align_ground_truth_fields([field], [description])
    assert rejected.matches == []
    assert rejected.unmatched == [field]

    accepted = align_ground_truth_fields(
        [field],
        [description, _region("vat", "5%", 300, 10)],
    )
    match = accepted.matches[0]
    assert match.candidate.regions[0].region_id == "vat"
    assert match.candidate_count == 1
    assert match.training_eligible is True


def test_unmatched_fields_remain_in_alignment_result() -> None:
    field = _field("missing", "INVOICE_LOOKUP_CODE", "NOT-PRESENT", kind="identifier")
    result = align_ground_truth_fields([field], [_region("other", "DIFFERENT", 10, 10)])
    assert result.matches == []
    assert result.unmatched == [field]


def test_date_matching_rejects_date_embedded_in_description_text() -> None:
    field = _field(
        "expiry",
        "EXPIRY_DATE",
        "2026-01-02",
        kind="date",
        item_index=0,
    )
    description = _region(
        "description",
        "Synthetic lot L1 expiry 02/01/2026",
        10,
        10,
    )

    rejected = align_ground_truth_fields([field], [description])
    assert rejected.matches == []

    accepted = align_ground_truth_fields(
        [field],
        [description, _region("expiry", "02/01/2026", 300, 10)],
    )
    match = accepted.matches[0]
    assert match.candidate.regions[0].region_id == "expiry"
    assert match.training_eligible is True


def test_redundant_spans_around_one_date_are_not_duplicate_candidates() -> None:
    field = _field(
        "date",
        "INVOICE_DATE",
        "2026-01-02",
        kind="date",
    )
    regions = [
        _region("prefix", "Synthetic date", 10, 10),
        _region("date", "02/01/2026", 10, 20),
        _region("suffix", "Synthetic note", 10, 30),
    ]

    result = align_ground_truth_fields([field], regions)
    match = result.matches[0]
    assert [region.region_id for region in match.candidate.regions] == ["date"]
    assert match.candidate_count == 1
    assert match.duplicate_candidates is False
    assert match.training_eligible is True


def test_repeated_item_values_resolve_by_row_geometry_and_order() -> None:
    fields = [
        _field("item0.name", "MEDICINE_NAME", "Medicine Alpha", item_index=0),
        _field("item0.qty", "QUANTITY", "5", kind="number", item_index=0),
        _field("item1.name", "MEDICINE_NAME", "Medicine Beta", item_index=1),
        _field("item1.qty", "QUANTITY", "5", kind="number", item_index=1),
    ]
    regions = [
        _region("name-0", "Medicine Alpha", 20, 100),
        _region("qty-0", "5", 300, 100),
        _region("name-1", "Medicine Beta", 20, 200),
        _region("qty-1", "5", 300, 200),
    ]
    result = align_ground_truth_fields(fields, regions)
    matches = {match.field.field_path: match for match in result.matches}
    assert matches["item0.qty"].candidate.regions[0].region_id == "qty-0"
    assert matches["item1.qty"].candidate.regions[0].region_id == "qty-1"
    assert matches["item0.qty"].ambiguous is False
    assert matches["item1.qty"].ambiguous is False


def test_layout_bbox_normalization_uses_zero_to_one_thousand_convention() -> None:
    box = BoundingBox(x_min=10, y_min=20, x_max=50, y_max=80)
    assert normalize_layout_bbox(box, width=100, height=100) == [100, 200, 500, 800]
