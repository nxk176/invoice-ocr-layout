"""Layout/KIE adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from invoice_ocr.contracts import (
    DocumentPage,
    LabeledEntity,
    RecognizedRegion,
    Relation,
)
from invoice_ocr.model_manifest import adapter_revision


class LayoutAdapter(ABC):
    """A layout model labels OCR regions and optionally predicts relations."""

    name: str

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

    @abstractmethod
    def extract(
        self, page: DocumentPage, regions: list[RecognizedRegion]
    ) -> tuple[list[LabeledEntity], list[Relation]]:
        """Predict invoice entities and relations."""
