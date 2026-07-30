"""PaddleOCR text detector adapter."""

from __future__ import annotations

from typing import Any

from invoice_ocr.adapters.detectors.base import DetectorAdapter
from invoice_ocr.contracts import (
    BoundingBox,
    DetectionRegion,
    DocumentPage,
    Point,
    ProcessingStatus,
)
from invoice_ocr.exceptions import CheckpointUnavailableError, DependencyUnavailableError


class PaddleOCRDetector(DetectorAdapter):
    name = "paddleocr"
    inference_implementation_available = True

    def _resolve_checkpoint(self) -> Any:
        checkpoint = self.checkpoint or self.model_root / "paddleocr" / "detector"
        required = ("inference.pdmodel", "inference.pdiparams")
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise CheckpointUnavailableError(
                f"PaddleOCR detector checkpoint is incomplete at {checkpoint}; missing "
                f"{', '.join(missing)}. Run 'python scripts/download_models.py "
                "--model paddleocr-detector'."
            )
        return checkpoint

    def _create_engine(self) -> Any:
        existing = getattr(self, "_engine", None)
        if existing is not None:
            return existing
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise DependencyUnavailableError(
                "PaddleOCR detector requires the 'paddleocr' package and a compatible "
                "PaddlePaddle build. Install the paddleocr extra, then install paddlepaddle "
                "or paddlepaddle-gpu following the official compatibility table."
            ) from exc
        kwargs: dict[str, Any] = {
            "use_angle_cls": False,
            "use_gpu": self.device == "cuda",
            "show_log": False,
            "det_model_dir": str(self._resolve_checkpoint()),
        }
        self._engine = PaddleOCR(**kwargs)
        return self._engine

    def prepare(self) -> None:
        self._create_engine()

    def detect(self, page: DocumentPage) -> list[DetectionRegion]:
        engine = self._create_engine()
        raw_result = engine.ocr(page.image_path, det=True, rec=False, cls=False)
        polygons = raw_result[0] if raw_result and raw_result[0] else []
        regions: list[DetectionRegion] = []
        for index, raw_polygon in enumerate(polygons):
            points = [Point(x=float(point[0]), y=float(point[1])) for point in raw_polygon]
            x_values = [point.x for point in points]
            y_values = [point.y for point in points]
            regions.append(
                DetectionRegion(
                    document_id=page.document_id,
                    source_path=page.source_path,
                    page_index=page.page_index,
                    model_name=self.name,
                    model_revision=self.revision,
                    processing_status=ProcessingStatus.SUCCESS,
                    region_id=f"{page.document_id}-p{page.page_index}-r{index}",
                    polygon=points,
                    bbox=BoundingBox(
                        x_min=min(x_values),
                        y_min=min(y_values),
                        x_max=max(x_values),
                        y_max=max(y_values),
                    ),
                    confidence=1.0,
                )
            )
        return regions
