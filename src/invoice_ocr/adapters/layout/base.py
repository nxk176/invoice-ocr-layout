"""Layout/KIE adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from invoice_ocr.contracts import (
    DocumentPage,
    LabeledEntity,
    PretrainedLayoutTrace,
    RecognizedRegion,
    Relation,
)
from invoice_ocr.model_manifest import adapter_revision


class LayoutAdapter(ABC):
    """A layout model labels OCR regions and optionally predicts relations."""

    name: str
    inference_implementation_available: bool = False
    provides_invoice_labels: bool = True

    def __init__(
        self,
        model_root: Path,
        device: str = "auto",
        checkpoint: Path | None = None,
        revision: str | None = None,
    ) -> None:
        self.model_root = model_root
        self.device = device
        self.checkpoint = checkpoint
        try:
            manifest_revision = adapter_revision("layout", self.name)
        except KeyError:
            manifest_revision = None
        self.revision = revision or manifest_revision

    def prepare(self) -> None:
        """Load and validate reusable runtime state before timed inference."""
        return None

    def raw_trace(self) -> PretrainedLayoutTrace | None:
        """Return evidence for the latest label-free pretrained forward pass."""
        return None

    @abstractmethod
    def extract(
        self, page: DocumentPage, regions: list[RecognizedRegion]
    ) -> tuple[list[LabeledEntity], list[Relation]]:
        """Predict invoice entities and relations."""
