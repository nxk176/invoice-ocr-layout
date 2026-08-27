from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from PIL import Image

from invoice_ocr.adapters.detectors.base import DetectorAdapter
from invoice_ocr.adapters.recognizers.base import RecognizerAdapter
from invoice_ocr.contracts import (
    BoundingBox,
    DetectionRegion,
    DocumentPage,
    InvoiceBatch,
    InvoiceDocument,
    InvoiceHeader,
    Point,
    ProcessingStatus,
    RecognizedRegion,
)
from invoice_ocr.io.paths import discover_documents
from invoice_ocr.layout_gt.builder import (
    LayoutGTBuildRequest,
    LayoutGTRealignRequest,
    build_layout_ground_truth,
    inspect_layout_ground_truth,
    realign_layout_ground_truth,
)


class SyntheticDetector(DetectorAdapter):
    name = "synthetic-detector"

    def detect(self, page: DocumentPage) -> list[DetectionRegion]:
        box = BoundingBox(x_min=10, y_min=10, x_max=70, y_max=30)
        polygon = [
            Point(x=box.x_min, y=box.y_min),
            Point(x=box.x_max, y=box.y_min),
            Point(x=box.x_max, y=box.y_max),
            Point(x=box.x_min, y=box.y_max),
        ]
        return [
            DetectionRegion(
                document_id=page.document_id,
                source_path=page.source_path,
                page_index=page.page_index,
                model_name=self.name,
                processing_status=ProcessingStatus.SUCCESS,
                region_id=f"{page.document_id}-number",
                polygon=polygon,
                bbox=box,
                confidence=0.95,
            )
        ]


class SyntheticRecognizer(RecognizerAdapter):
    name = "synthetic-recognizer"

    def recognize(
        self,
        page: DocumentPage,
        regions: list[DetectionRegion],
    ) -> list[RecognizedRegion]:
        return [
            RecognizedRegion(
                **region.model_dump(
                    exclude={"confidence", "model_name", "model_revision"},
                ),
                model_name=self.name,
                model_revision=None,
                text="SYN-001",
                confidence=0.9,
            )
            for region in regions
        ]


def test_builder_serializes_trainable_pseudo_layout_dataset_and_reports_na_iou(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="invoice_ocr")
    data = tmp_path / "data" / "t5"
    source = data / "synthetic-folder" / "invoice.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), color="white").save(source)
    document = discover_documents(data)[0]
    gt_path = tmp_path / "GT" / "final" / "t5" / "synthetic-folder" / "invoice.json"
    gt_path.parent.mkdir(parents=True)
    payload = InvoiceBatch(
        invoice_count=1,
        invoices=[
            InvoiceDocument(
                page_number=1,
                invoice=InvoiceHeader(invoice_number="SYN-001"),
            )
        ],
    ).model_dump(mode="json")
    payload["_review"] = {"synthetic": True}
    gt_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "work" / "layout_gt" / "t5"

    build_layout_ground_truth(
        LayoutGTBuildRequest(
            input_root=data,
            gt_root=tmp_path / "GT",
            output_dir=output,
            detector_name="synthetic-detector",
            recognizer_name="synthetic-recognizer",
        ),
        detector=SyntheticDetector(tmp_path / "models", "cpu"),
        recognizer=SyntheticRecognizer(tmp_path / "models", "cpu"),
    )

    annotation_path = output / "layout" / f"{document.document_id}.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    page = annotation["pages"][0]
    assert page["tokens"] == ["SYN-001"]
    assert page["labels"] == ["B-INVOICE_NUMBER"]
    assert page["boxes"] == [[100, 100, 700, 300]]
    assert annotation["alignments"][0]["source"] == "pretrained_ocr_alignment"
    assert annotation["alignments"][0]["detection_confidence"] == 0.95
    assert annotation["alignments"][0]["recognition_confidence"] == 0.9

    report = json.loads((output / "alignment_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["matched_fields"] == 1
    assert report["summary"]["unmatched_fields"] == 0
    assert report["detector_iou"]["status"] == "N/A"
    inspected = inspect_layout_ground_truth(output)
    assert inspected["valid_layout_dataset"] is True
    assert inspected["layout_annotation_count"] == 1
    assert "[1/1] Processing synthetic-folder/invoice.png" in caplog.text
    assert "[1/1] Completed synthetic-folder/invoice.png" in caplog.text
    assert "Pseudo-layout GT build completed" in caplog.text

    ocr_path = output / "ocr" / f"{document.document_id}.json"
    image_path = output / page["image_path"]
    protected_paths = (source, gt_path, ocr_path, image_path)
    protected_contents = {path: path.read_bytes() for path in protected_paths}

    realign_layout_ground_truth(LayoutGTRealignRequest(layout_gt_root=output))

    assert {path: path.read_bytes() for path in protected_paths} == protected_contents
    realigned = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert realigned["pages"][0]["labels"] == ["B-INVOICE_NUMBER"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["realignment_count"] == 1
    assert manifest["last_realignment_source"] == "cached_pretrained_ocr"
    assert "Cached pseudo-layout realignment completed" in caplog.text
