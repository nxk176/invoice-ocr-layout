"""Fine-tuned LayoutLMv3 token-classification adapter."""

from __future__ import annotations

from typing import Any

from invoice_ocr.adapters.layout.base import LayoutAdapter
from invoice_ocr.contracts import DocumentPage, LabeledEntity, RecognizedRegion, Relation
from invoice_ocr.exceptions import CheckpointUnavailableError, DependencyUnavailableError


class LayoutLMv3Adapter(LayoutAdapter):
    name = "layoutlmv3"

    def _load(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoProcessor
        except ImportError as exc:
            raise DependencyUnavailableError(
                "LayoutLMv3 requires torch and transformers. Install the layoutlmv3 extra "
                "with a PyTorch build compatible with the selected CPU/CUDA runtime."
            ) from exc
        checkpoint = self.checkpoint or self.model_root / "layoutlmv3" / "invoice-best"
        if not checkpoint.is_dir():
            raise CheckpointUnavailableError(
                f"fine-tuned LayoutLMv3 checkpoint not found at {checkpoint}. A base "
                "checkpoint does not know invoice labels; run layout training first."
            )
        processor = AutoProcessor.from_pretrained(checkpoint, apply_ocr=False)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        device = "cuda" if self.device == "cuda" else "cpu"
        model.to(device)
        model.eval()
        return torch, processor, model

    def extract(
        self, page: DocumentPage, regions: list[RecognizedRegion]
    ) -> tuple[list[LabeledEntity], list[Relation]]:
        torch, processor, model = self._load()
        words = [region.text for region in regions]
        boxes = [
            [
                round(region.bbox.normalize(page.width, page.height).x_min),
                round(region.bbox.normalize(page.width, page.height).y_min),
                round(region.bbox.normalize(page.width, page.height).x_max),
                round(region.bbox.normalize(page.width, page.height).y_max),
            ]
            for region in regions
        ]
        encoded = processor(
            images=page.image_path,
            text=words,
            boxes=boxes,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        if logits.shape[0] != 1:
            raise RuntimeError("LayoutLMv3 adapter expected a single-page batch")
        raise RuntimeError(
            "LayoutLMv3 word-id alignment depends on the saved processor tokenizer. "
            "Use the training-produced inference metadata; this checkpoint lacks a supported "
            "alignment contract."
        )
