"""Detector adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from invoice_ocr.contracts import DetectionRegion, DocumentPage
from invoice_ocr.model_manifest import adapter_revision


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
        try:
            manifest_revision = adapter_revision("detector", self.name)
        except KeyError:
            manifest_revision = None
        self.revision = revision or manifest_revision

    def prepare(self) -> None:
        """Load and validate reusable runtime state before timed inference."""
        return None

    @abstractmethod
    def detect(self, page: DocumentPage) -> list[DetectionRegion]:
        """Detect text polygons on one rendered page."""
