"""Recognizer adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from invoice_ocr.contracts import DetectionRegion, DocumentPage, RecognizedRegion


class RecognizerAdapter(ABC):
    """A recognizer transcribes detector regions without changing their geometry."""

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
    def recognize(
        self, page: DocumentPage, regions: list[DetectionRegion]
    ) -> list[RecognizedRegion]:
        """Transcribe detected regions on one page."""
