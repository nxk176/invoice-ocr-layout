from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from invoice_ocr.layout_gt.builder import inspect_layout_ground_truth
from invoice_ocr.layout_gt.cleaner import LayoutGTCleanRequest, clean_layout_ground_truth
from invoice_ocr.training.datasets import load_layout_pages
from invoice_ocr.training.layout import LayoutPageDataset


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _alignment(
    field_path: str,
    label: str,
    region_ids: list[str],
    *,
    method: str = "exact",
    ambiguous: bool = False,
    duplicate: bool = False,
) -> dict[str, Any]:
    return {
        "field_path": field_path,
        "label": label,
        "gt_value": "SYNTHETIC",
        "ocr_text": "SYNTHETIC",
        "region_ids": region_ids,
        "match_method": method,
        "match_confidence": 1.0,
        "recognition_confidence": 0.99,
        "detection_confidence": 0.99,
        "ambiguous": ambiguous,
        "duplicate_candidates": duplicate,
        "candidate_count": 2 if duplicate else 1,
        "training_eligible": not ambiguous,
    }


def _source_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    image = source / "images" / "page.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="white").save(image)
    _write_json(
        source / "manifest.json",
        {
            "schema_version": "pseudo-layout-gt-v1",
            "input_root": str(tmp_path / "data"),
            "gt_root": str(tmp_path / "GT"),
        },
    )
    annotation = source / "layout" / "synthetic.json"
    _write_json(
        annotation,
        {
            "document_id": "synthetic",
            "annotation_kind": "pseudo_layout_gt",
            "pages": [
                {
                    "page_index": 0,
                    "image_path": "images/page.png",
                    "tokens": ["SYNTHETIC", "FUZZY", "BACKGROUND", "CONFLICT"],
                    "boxes": [
                        [0, 0, 100, 100],
                        [100, 0, 200, 100],
                        [200, 0, 300, 100],
                        [300, 0, 400, 100],
                    ],
                    "labels": ["B-OLD", "O", "O", "B-OLD"],
                    "region_ids": ["safe", "fuzzy", "background", "conflict"],
                    "regions": [
                        {"region_id": "safe", "label": "B-OLD"},
                        {"region_id": "fuzzy", "label": "O"},
                        {"region_id": "background", "label": "O"},
                        {"region_id": "conflict", "label": "B-OLD"},
                    ],
                }
            ],
            "alignments": [
                _alignment("invoice.number", "INVOICE_NUMBER", ["safe"]),
                _alignment("supplier.name", "SUPPLIER_NAME", ["fuzzy"], method="fuzzy"),
                _alignment("totals.subtotal", "SUBTOTAL", ["conflict"]),
                _alignment("items.0.amount", "LINE_AMOUNT", ["conflict"]),
            ],
            "unmatched_fields": [
                {"field_path": "buyer.name", "label": "BUYER_ORGANIZATION", "gt_value": "X"}
            ],
        },
    )
    return source, annotation


def test_conservative_cleaner_masks_uncertain_regions_and_preserves_source(tmp_path: Path) -> None:
    source, annotation = _source_dataset(tmp_path)
    before = annotation.read_bytes()
    output = tmp_path / "clean"

    clean_layout_ground_truth(LayoutGTCleanRequest(source, output))

    assert annotation.read_bytes() == before
    cleaned = json.loads((output / "layout" / annotation.name).read_text(encoding="utf-8"))
    page = cleaned["pages"][0]
    assert page["labels"] == ["B-INVOICE_NUMBER", "O", "O", "O"]
    assert page["ignore_mask"] == [False, True, False, True]
    assert (output / page["image_path"]).resolve().is_file()
    decisions = {row["field_path"]: row["cleaning_decision"] for row in cleaned["alignments"]}
    assert decisions == {
        "invoice.number": "KEEP",
        "supplier.name": "REVIEW",
        "totals.subtotal": "IGNORE",
        "items.0.amount": "IGNORE",
    }
    assert cleaned["unmatched_fields"][0]["cleaning_decision"] == "IGNORE"
    report = json.loads((output / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["keep_fields"] == 1
    assert report["summary"]["review_fields"] == 1
    assert report["summary"]["ignore_fields"] == 3
    assert report["decisions_by_field"]["INVOICE_NUMBER"]["keep_fields"] == 1
    queue = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))
    assert queue["review_count"] == 1
    inspection = inspect_layout_ground_truth(output)
    assert inspection["valid_layout_dataset"] is True
    assert inspection["report_kind"] == "cleaning_report"
    assert inspection["summary"]["keep_fields"] == 1


class _FakeEncoding(dict[str, Any]):
    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        return [None, 0, 1, 2, 3, None]


class _FakeProcessor:
    def __call__(self, **_kwargs: Any) -> _FakeEncoding:
        return _FakeEncoding(input_ids=[0, 1, 2, 3, 4, 5])


def test_cleaned_ignore_mask_becomes_minus_100_in_layout_training(tmp_path: Path) -> None:
    source, _ = _source_dataset(tmp_path)
    output = tmp_path / "clean"
    clean_layout_ground_truth(LayoutGTCleanRequest(source, output))
    annotation = output / "layout" / "synthetic.json"
    pages = load_layout_pages([annotation])
    dataset = LayoutPageDataset(
        pages,
        _FakeProcessor(),
        {"O": 0, "B-INVOICE_NUMBER": 1},
        output,
    )

    assert dataset[0]["labels"] == [-100, 1, -100, 0, -100, -100]
