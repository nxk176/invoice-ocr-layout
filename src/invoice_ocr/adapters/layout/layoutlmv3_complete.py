"""Complete fine-tuned LayoutLMv3 token-classification runtime."""

from __future__ import annotations

from typing import Any

from PIL import Image

from invoice_ocr.adapters.layout.base import LayoutAdapter
from invoice_ocr.contracts import (
    DocumentPage,
    LabeledEntity,
    ProcessingStatus,
    RecognizedRegion,
    Relation,
)
from invoice_ocr.exceptions import CheckpointUnavailableError, DependencyUnavailableError


class LayoutLMv3Adapter(LayoutAdapter):
    name = "layoutlmv3"

    def resolve_checkpoint(self) -> Any:
        candidates = [
            self.checkpoint,
            self.model_root / "layoutlmv3" / "invoice-best",
            self.model_root.parent / "outputs" / "training" / "layout-layoutlmv3" / "invoice-best",
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_dir():
                return candidate
        expected = ", ".join(str(path) for path in candidates if path is not None)
        raise CheckpointUnavailableError(
            "fine-tuned LayoutLMv3 checkpoint not found. Checked: "
            f"{expected}. A base checkpoint does not know invoice labels; run layout "
            "training first or place invoice-best under models/layoutlmv3."
        )

    def _load(self) -> tuple[Any, Any, Any]:
        existing: tuple[Any, Any, Any] | None = getattr(self, "_runtime", None)
        if existing is not None:
            return existing
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoProcessor
        except ImportError as exc:
            raise DependencyUnavailableError(
                "LayoutLMv3 requires torch and transformers. Install the layoutlmv3 extra "
                "with a PyTorch build compatible with the selected CPU/CUDA runtime."
            ) from exc
        checkpoint = self.resolve_checkpoint()
        processor = AutoProcessor.from_pretrained(checkpoint, apply_ocr=False)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        model.to("cuda" if self.device == "cuda" else "cpu")
        model.eval()
        self._runtime = (torch, processor, model)
        return self._runtime

    def prepare(self) -> None:
        self._load()

    def extract(
        self, page: DocumentPage, regions: list[RecognizedRegion]
    ) -> tuple[list[LabeledEntity], list[Relation]]:
        torch, processor, model = self._load()
        normalized = [region.bbox.normalize(page.width, page.height) for region in regions]
        boxes = [
            [round(box.x_min), round(box.y_min), round(box.x_max), round(box.y_max)]
            for box in normalized
        ]
        with Image.open(page.image_path) as source:
            encoding = processor(
                images=source.convert("RGB"),
                text=[region.text for region in regions],
                boxes=boxes,
                truncation=True,
                return_tensors="pt",
            )
        word_ids = encoding.word_ids(batch_index=0)
        model_inputs = {key: value.to(model.device) for key, value in encoding.items()}
        with torch.no_grad():
            probabilities = torch.softmax(model(**model_inputs).logits[0], dim=-1)
        entities: list[LabeledEntity] = []
        seen_words: set[int] = set()
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_words or word_id >= len(regions):
                continue
            seen_words.add(word_id)
            confidence, label_id = probabilities[token_index].max(dim=-1)
            label = str(model.config.id2label[int(label_id)])
            if label == "O":
                continue
            region = regions[word_id]
            entities.append(
                LabeledEntity(
                    document_id=page.document_id,
                    source_path=page.source_path,
                    page_index=page.page_index,
                    model_name=self.name,
                    model_revision=self.revision,
                    processing_status=ProcessingStatus.SUCCESS,
                    entity_id=f"{region.region_id}-{label}",
                    label=label,
                    text=region.text,
                    bbox=region.bbox,
                    confidence=float(confidence),
                    region_ids=[region.region_id],
                )
            )
        return entities, []
