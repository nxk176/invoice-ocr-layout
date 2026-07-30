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
from invoice_ocr.exceptions import CheckpointUnavailableError, DependencyUnavailableError


class PaddleOCRRecognizer(RecognizerAdapter):
    name = "paddleocr"
    inference_implementation_available = True

    def _resolve_checkpoint(self) -> Any:
        checkpoint = self.checkpoint or self.model_root / "paddleocr" / "recognizer"
        required = ("inference.pdmodel", "inference.pdiparams")
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise CheckpointUnavailableError(
                f"PaddleOCR recognizer checkpoint is incomplete at {checkpoint}; missing "
                f"{', '.join(missing)}. Run 'python scripts/download_models.py "
                "--model paddleocr-recognizer'."
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
                "PaddleOCR recognizer requires the 'paddleocr' package and a compatible "
                "PaddlePaddle build. Follow the official CPU/CUDA installation matrix."
            ) from exc
        kwargs: dict[str, Any] = {
            "det": False,
            "use_angle_cls": False,
            "use_gpu": self.device == "cuda",
            "show_log": False,
            "lang": "vi",
            "rec_model_dir": str(self._resolve_checkpoint()),
        }
        self._engine = PaddleOCR(**kwargs)
        return self._engine

    def prepare(self) -> None:
        self._create_engine()

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
