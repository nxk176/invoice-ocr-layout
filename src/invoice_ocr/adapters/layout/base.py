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
        self.revision = revision

    @abstractmethod
    def extract(
        self, page: DocumentPage, regions: list[RecognizedRegion]
    ) -> tuple[list[LabeledEntity], list[Relation]]:
        """Predict invoice entities and relations."""

