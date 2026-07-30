"""PaddleOCR recognition-only adapter."""

from __future__ import annotations

from typing import Any

from PIL import Image

from invoice_ocr.adapters.recognizers.base import RecognizerAdapter
from invoice_ocr.contracts import (
    DetectionRegion,
    DocumentPage,
    ProcessingStatus,
    RecognizedRegion,
)
from invoice_ocr.exceptions import DependencyUnavailableError


class PaddleOCRRecognizer(RecognizerAdapter):
    name = "paddleocr"

    def _create_engine(self) -> Any:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise DependencyUnavailableError(
                "PaddleOCR recognizer requires the 'paddleocr' package and a compatible "
                "PaddlePaddle build. Follow the official CPU/CUDA installation matrix."
            ) from exc
        kwargs: dict[str, Any] = {
            "det": False,
            "use_angle_cls": False,
            "use_gpu": self.device == "cuda",
            "show_log": False,
            "lang": "vi",
        }
        if self.checkpoint is not None:
            kwargs["rec_model_dir"] = str(self.checkpoint)
        return PaddleOCR(**kwargs)

    def recognize(
        self, page: DocumentPage, regions: list[DetectionRegion]
    ) -> list[RecognizedRegion]:
        engine = self._create_engine()
        image = Image.open(page.image_path).convert("RGB")
        results: list[RecognizedRegion] = []
        for region in regions:
            box = region.bbox
            crop = image.crop((box.x_min, box.y_min, box.x_max, box.y_max))
            raw = engine.ocr(crop, det=False, rec=True, cls=False)
            candidate = raw[0][0] if raw and raw[0] else ("", 0.0)
            text, confidence = str(candidate[0]), float(candidate[1])
            results.append(
                RecognizedRegion(
                    document_id=page.document_id,
                    source_path=page.source_path,
                    page_index=page.page_index,
                    model_name=self.name,
                    model_revision=self.revision,
                    processing_status=ProcessingStatus.SUCCESS,
                    region_id=region.region_id,
                    polygon=region.polygon,
                    bbox=region.bbox,
                    text=text,
                    confidence=confidence,
                )
            )
        return results

