"""VietOCR region recognizer adapter."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
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


class VietOCRRecognizer(RecognizerAdapter):
    name = "vietocr"
    inference_implementation_available = True

    def _create_predictor(self) -> Any:
        existing = getattr(self, "_predictor", None)
        if existing is not None:
            return existing
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise DependencyUnavailableError(
                "VietOCR requires the 'vietocr' package. Install the vietocr extra and a "
                "compatible PyTorch build."
            ) from exc
        checkpoint = self.checkpoint or self.model_root / "vietocr" / "transformerocr.pth"
        if checkpoint.is_dir():
            weight_files = sorted(checkpoint.glob("*.pth"))
            if len(weight_files) == 1:
                checkpoint = weight_files[0]
        if not checkpoint.is_file():
            raise CheckpointUnavailableError(
                f"VietOCR checkpoint not found at {checkpoint}. Run "
                "'python scripts/download_models.py --model vietocr' or pass a trained checkpoint."
            )
        config = Cfg.load_config_from_name("vgg_transformer")
        config["weights"] = str(checkpoint)
        config["device"] = "cuda:0" if self.device == "cuda" else "cpu"
        config["cnn"]["pretrained"] = False
        self._predictor = Predictor(config)
        return self._predictor

    def prepare(self) -> None:
        self._create_predictor()

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        """Keep invalid backend probabilities from corrupting the JSONL contract."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return confidence if isfinite(confidence) and 0.0 <= confidence <= 1.0 else 0.0

    @staticmethod
    def _crop(image: Image.Image, region: DetectionRegion) -> Image.Image:
        box = region.bbox
        return image.crop((box.x_min, box.y_min, box.x_max, box.y_max))

    def recognize(
        self, page: DocumentPage, regions: list[DetectionRegion]
    ) -> list[RecognizedRegion]:
        predictor = self._create_predictor()
        image = Image.open(Path(page.image_path)).convert("RGB")
        results: list[RecognizedRegion] = []
        for region in regions:
            text, confidence = predictor.predict(self._crop(image, region), return_prob=True)
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
                    text=str(text),
                    confidence=self._safe_confidence(confidence),
                )
            )
        return results
