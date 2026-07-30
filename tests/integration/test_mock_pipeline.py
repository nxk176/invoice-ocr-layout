from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from invoice_ocr.adapters.detectors.base import DetectorAdapter
from invoice_ocr.adapters.layout.base import LayoutAdapter
from invoice_ocr.adapters.recognizers.base import RecognizerAdapter
from invoice_ocr.contracts import (
    BoundingBox,
    DetectionRegion,
    DocumentPage,
    LabeledEntity,
    Point,
    ProcessingStatus,
    RecognizedRegion,
    Relation,
)
from invoice_ocr.pipeline import (
    PipelineOptions,
    PipelineRunner,
    PipelineSelection,
    validate_canonical_payload,
)


class MockDetector(DetectorAdapter):
    name = "mock-detector"
    calls = 0

    def detect(self, page: DocumentPage) -> list[DetectionRegion]:
        self.calls += 1
        boxes = [
            BoundingBox(x_min=10, y_min=10, x_max=100, y_max=30),
            BoundingBox(x_min=10, y_min=40, x_max=100, y_max=60),
            BoundingBox(x_min=10, y_min=100, x_max=120, y_max=120),
            BoundingBox(x_min=140, y_min=100, x_max=190, y_max=120),
            BoundingBox(x_min=210, y_min=100, x_max=270, y_max=120),
            BoundingBox(x_min=10, y_min=140, x_max=120, y_max=160),
            BoundingBox(x_min=140, y_min=140, x_max=190, y_max=160),
            BoundingBox(x_min=210, y_min=140, x_max=270, y_max=160),
            BoundingBox(x_min=10, y_min=200, x_max=100, y_max=220),
            BoundingBox(x_min=140, y_min=200, x_max=220, y_max=220),
        ]
        return [
            DetectionRegion(
                document_id=page.document_id,
                source_path=page.source_path,
                page_index=page.page_index,
                model_name=self.name,
                model_revision="test",
                processing_status=ProcessingStatus.SUCCESS,
                region_id=f"r{index}",
                polygon=[
                    Point(x=box.x_min, y=box.y_min),
                    Point(x=box.x_max, y=box.y_min),
                    Point(x=box.x_max, y=box.y_max),
                    Point(x=box.x_min, y=box.y_max),
                ],
                bbox=box,
                confidence=1,
            )
            for index, box in enumerate(boxes)
        ]


class MockRecognizer(RecognizerAdapter):
    name = "mock-recognizer"

    def recognize(
        self, page: DocumentPage, regions: list[DetectionRegion]
    ) -> list[RecognizedRegion]:
        texts = [
            "000042",
            "02/01/2026",
            "Synthetic Med A",
            "0007",
            "60",
            "Synthetic Med B",
            "0008",
            "40",
            "100",
            "110",
        ]
        return [
            RecognizedRegion(
                document_id=page.document_id,
                source_path=page.source_path,
                page_index=page.page_index,
                model_name=self.name,
                model_revision="test",
                processing_status=ProcessingStatus.SUCCESS,
                region_id=region.region_id,
                polygon=region.polygon,
                bbox=region.bbox,
                text=text,
                confidence=1,
            )
            for region, text in zip(regions, texts, strict=False)
        ]


class MockLayout(LayoutAdapter):
    name = "mock-layout"

    def extract(
        self, page: DocumentPage, regions: list[RecognizedRegion]
    ) -> tuple[list[LabeledEntity], list[Relation]]:
        labels = [
            "INVOICE_NUMBER",
            "INVOICE_DATE",
            "RAW_DESCRIPTION",
            "LOT_NUMBER",
            "LINE_AMOUNT",
            "RAW_DESCRIPTION",
            "LOT_NUMBER",
            "LINE_AMOUNT",
            "SUBTOTAL",
            "GRAND_TOTAL",
        ]
        entities = [
            LabeledEntity(
                document_id=page.document_id,
                source_path=page.source_path,
                page_index=page.page_index,
                model_name=self.name,
                model_revision="test",
                processing_status=ProcessingStatus.SUCCESS,
                entity_id=f"e{index}",
                label=label,
                text=region.text,
                bbox=region.bbox,
                confidence=1,
                region_ids=[region.region_id],
            )
            for index, (region, label) in enumerate(zip(regions, labels, strict=False))
        ]
        return entities, []


def test_synthetic_mock_pipeline_and_resume(tmp_path: Path) -> None:
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    image_path = input_dir / "synthetic_invoice.png"
    Image.new("RGB", (400, 300), "white").save(image_path)
    output = tmp_path / "outputs" / "smoke"
    options = PipelineOptions(
        input_path=input_dir,
        output_path=output,
        work_root=tmp_path / "work",
        model_root=tmp_path / "models",
        workflow_defaults=Path("configs/workflow_defaults/default.yaml"),
        device="cpu",
        resume=True,
        keep_intermediate=True,
    )
    detector = MockDetector(tmp_path)
    runner = PipelineRunner(
        PipelineSelection("paddleocr", "vietocr", "layoutlmv3"),
        options,
        detector=detector,
        recognizer=MockRecognizer(tmp_path),
        layout=MockLayout(tmp_path),
    )
    manifest = runner.run()
    assert manifest.failed_document_count == 0
    predictions = list((output / "predictions").glob("*.json"))
    assert len(predictions) == 1
    payload = json.loads(predictions[0].read_text(encoding="utf-8"))
    validate_canonical_payload(payload)
    assert payload["invoices"][0]["invoice"]["invoice_number"] == "000042"
    assert [item["lot_number"] for item in payload["invoices"][0]["items"]] == [
        "0007",
        "0008",
    ]
    assert payload["invoices"][0]["validation"]["sum_of_items_equals_subtotal"] is True
    for filename in (
        "pages.jsonl",
        "detections.jsonl",
        "recognitions.jsonl",
        "entities.jsonl",
        "tables.jsonl",
    ):
        assert (tmp_path / "work" / "smoke" / filename).is_file()
    first_call_count = detector.calls
    runner.run()
    assert detector.calls == first_call_count
