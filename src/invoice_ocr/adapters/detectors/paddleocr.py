"""PaddleOCR text detector adapter."""

from __future__ import annotations

import importlib
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
        checkpoint = self._resolve_checkpoint()
        try:
            # Importing the package module registers the top-level tools package.
            paddleocr_module = importlib.import_module("paddleocr.paddleocr")
            predict_det = importlib.import_module("tools.infer.predict_det")
        except (ImportError, OSError) as exc:
            raise DependencyUnavailableError(
                "PaddleOCR detector requires the 'paddleocr' package and a compatible "
                "PaddlePaddle build. Install the paddleocr extra, then install paddlepaddle "
                "or paddlepaddle-gpu following the official compatibility table."
            ) from exc
        params = paddleocr_module.parse_args(mMain=False)
        params.det = True
        params.rec = False
        params.use_angle_cls = False
        params.use_gpu = self.device == "cuda"
        params.show_log = False
        params.det_model_dir = str(checkpoint)
        # The legacy inference optimizer can SIGILL in SelfAttentionFusePass.
        params.ir_optim = False
        params.use_tensorrt = False
        params.enable_mkldnn = False
        self._engine = predict_det.TextDetector(params)
        return self._engine

    def prepare(self) -> None:
        self._create_engine()

    def detect(self, page: DocumentPage) -> list[DetectionRegion]:
        engine = self._create_engine()
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise DependencyUnavailableError(
                "PaddleOCR detection requires NumPy and Pillow. Install the "
                "paddleocr project extra."
            ) from exc
        with Image.open(page.image_path) as page_image:
            rgb_image = np.asarray(page_image.convert("RGB"))
        bgr_image = np.ascontiguousarray(rgb_image[:, :, ::-1])
        raw_boxes, _elapsed = engine(bgr_image)
        polygons = raw_boxes if raw_boxes is not None else []
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
