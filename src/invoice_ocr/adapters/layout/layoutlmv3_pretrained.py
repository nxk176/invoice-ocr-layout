"""Raw LayoutLMv3-base runtime for an end-to-end pretrained smoke run."""

from __future__ import annotations

from typing import Any

from PIL import Image

from invoice_ocr.adapters.layout.base import LayoutAdapter
from invoice_ocr.contracts import (
    DocumentPage,
    LabeledEntity,
    PretrainedLayoutTrace,
    ProcessingStatus,
    RecognizedRegion,
    Relation,
)
from invoice_ocr.exceptions import CheckpointUnavailableError, DependencyUnavailableError


class LayoutLMv3PretrainedAdapter(LayoutAdapter):
    """Run the base encoder without pretending it has invoice field labels."""

    name = "layoutlmv3_pretrained"
    inference_implementation_available = True
    provides_invoice_labels = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runtime: tuple[Any, Any, Any] | None = None
        self._last_trace: PretrainedLayoutTrace | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise DependencyUnavailableError(
                "LayoutLMv3 requires torch and transformers. Install the layoutlmv3 extra "
                "with a PyTorch build compatible with the selected CPU/CUDA runtime."
            ) from exc
        checkpoint = self.checkpoint or self.model_root / "layoutlmv3-base"
        if not checkpoint.is_dir():
            raise CheckpointUnavailableError(
                f"LayoutLMv3 base checkpoint not found at {checkpoint}. Download "
                "layoutlmv3-base before running the pretrained pipeline."
            )
        processor = AutoProcessor.from_pretrained(checkpoint, apply_ocr=False)
        model = AutoModel.from_pretrained(checkpoint)
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
        if not regions:
            self._last_trace = PretrainedLayoutTrace(
                document_id=page.document_id,
                source_path=page.source_path,
                page_index=page.page_index,
                model_name=self.name,
                model_revision=self.revision,
                processing_status=ProcessingStatus.NOT_AVAILABLE,
                checkpoint=str(self.checkpoint or self.model_root / "layoutlmv3-base"),
                recognized_region_count=0,
                encoded_token_count=0,
                hidden_size=int(model.config.hidden_size),
                message="No recognized text regions were available for LayoutLMv3-base.",
            )
            return [], []
        boxes = []
        for region in regions:
            box = region.bbox.normalize(page.width, page.height)
            boxes.append([round(box.x_min), round(box.y_min), round(box.x_max), round(box.y_max)])
        with Image.open(page.image_path) as source:
            encoding = processor(
                images=source.convert("RGB"),
                text=[region.text for region in regions],
                boxes=boxes,
                truncation=True,
                return_tensors="pt",
            )
        inputs = {key: value.to(model.device) for key, value in encoding.items()}
        with torch.no_grad():
            hidden_states = model(**inputs).last_hidden_state
        self._last_trace = PretrainedLayoutTrace(
            document_id=page.document_id,
            source_path=page.source_path,
            page_index=page.page_index,
            model_name=self.name,
            model_revision=self.revision,
            processing_status=ProcessingStatus.NOT_AVAILABLE,
            checkpoint=str(self.checkpoint or self.model_root / "layoutlmv3-base"),
            recognized_region_count=len(regions),
            encoded_token_count=int(hidden_states.shape[1]),
            hidden_size=int(hidden_states.shape[2]),
            message=(
                "LayoutLMv3-base forward pass completed. This base encoder has no invoice "
                "field classifier, so no semantic invoice entities or final JSON are emitted."
            ),
        )
        return [], []

    def raw_trace(self) -> PretrainedLayoutTrace | None:
        return self._last_trace
