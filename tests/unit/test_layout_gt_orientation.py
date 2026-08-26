from __future__ import annotations

from pathlib import Path

from PIL import Image

from invoice_ocr.adapters.detectors.base import DetectorAdapter
from invoice_ocr.adapters.recognizers.base import RecognizerAdapter
from invoice_ocr.contracts import (
    BoundingBox,
    DetectionRegion,
    DocumentPage,
    Point,
    ProcessingStatus,
    RecognizedRegion,
)
from invoice_ocr.layout_gt.orientation import (
    auto_orient_page,
    vertical_region_fraction,
)


class SyntheticOrientationDetector(DetectorAdapter):
    name = "synthetic-orientation-detector"

    def detect(self, page: DocumentPage) -> list[DetectionRegion]:
        horizontal = page.width > page.height
        regions: list[DetectionRegion] = []
        for index in range(8):
            if horizontal:
                box = BoundingBox(
                    x_min=5,
                    y_min=5 + index * 6,
                    x_max=35,
                    y_max=8 + index * 6,
                )
            else:
                box = BoundingBox(
                    x_min=5 + index * 6,
                    y_min=5,
                    x_max=8 + index * 6,
                    y_max=35,
                )
            regions.append(
                DetectionRegion(
                    document_id=page.document_id,
                    source_path=page.source_path,
                    page_index=page.page_index,
                    model_name=self.name,
                    processing_status=ProcessingStatus.SUCCESS,
                    region_id=f"{page.document_id}-r{index}",
                    polygon=[
                        Point(x=box.x_min, y=box.y_min),
                        Point(x=box.x_max, y=box.y_min),
                        Point(x=box.x_max, y=box.y_max),
                        Point(x=box.x_min, y=box.y_max),
                    ],
                    bbox=box,
                    confidence=0.9,
                )
            )
        return regions


class SyntheticOrientationRecognizer(RecognizerAdapter):
    name = "synthetic-orientation-recognizer"

    def __init__(self, model_root: Path) -> None:
        super().__init__(model_root, "cpu")
        self.calls = 0

    def recognize(
        self,
        page: DocumentPage,
        regions: list[DetectionRegion],
    ) -> list[RecognizedRegion]:
        self.calls += 1
        upright = page.orientation.rotation_degrees == 90
        return [
            RecognizedRegion(
                **region.model_dump(
                    exclude={"confidence", "model_name", "model_revision"},
                ),
                model_name=self.name,
                text="Synthetic readable text" if upright else "x",
                confidence=0.95 if upright else 0.1,
            )
            for region in regions
        ]


def _page(path: Path, width: int, height: int) -> DocumentPage:
    return DocumentPage(
        document_id="synthetic-page",
        source_path="synthetic.pdf",
        page_index=0,
        model_name="synthetic-renderer",
        processing_status=ProcessingStatus.SUCCESS,
        image_path=str(path),
        width=width,
        height=height,
    )


def test_auto_orientation_rotates_vertical_page_using_ocr_score(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (60, 120), color="white").save(image_path)
    page = _page(image_path, 60, 120)
    detector = SyntheticOrientationDetector(tmp_path)
    recognizer = SyntheticOrientationRecognizer(tmp_path)

    oriented, detections = auto_orient_page(page, detector, recognizer)

    assert oriented.orientation.rotation_degrees == 90
    assert oriented.orientation.method == "pretrained_ocr_orientation"
    assert oriented.width == 120
    assert oriented.height == 60
    assert recognizer.calls == 2
    assert vertical_region_fraction(detections) == 0.0
    with Image.open(image_path) as image:
        assert image.size == (120, 60)
    assert list(tmp_path.glob("*.orientation-*.png")) == []


def test_auto_orientation_leaves_horizontal_page_unchanged(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (120, 60), color="white").save(image_path)
    page = _page(image_path, 120, 60)
    detector = SyntheticOrientationDetector(tmp_path)
    recognizer = SyntheticOrientationRecognizer(tmp_path)

    oriented, detections = auto_orient_page(page, detector, recognizer)

    assert oriented == page
    assert recognizer.calls == 0
    assert vertical_region_fraction(detections) == 0.0
