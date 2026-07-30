"""Detector adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from invoice_ocr.contracts import DetectionRegion, DocumentPage


class DetectorAdapter(ABC):
    """A detector produces polygons only; it must not perform recognition."""

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
    def detect(self, page: DocumentPage) -> list[DetectionRegion]:
        """Detect text polygons on one rendered page."""

