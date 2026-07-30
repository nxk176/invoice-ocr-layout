"""Official DBNet repository adapter with explicit dependency checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from invoice_ocr.adapters.detectors.base import DetectorAdapter
from invoice_ocr.contracts import DetectionRegion, DocumentPage
from invoice_ocr.exceptions import CheckpointUnavailableError, DependencyUnavailableError


class DBNetDetector(DetectorAdapter):
    """Bridge for the official MhLiao/DB inference code.

    The official repository is not a stable Python package. We deliberately require an
    external revision-pinned checkout instead of vendoring or pretending that another
    implementation is equivalent.
    """

    name = "dbnet"
    algorithm = "DB"

    def _validate_runtime(self) -> Path:
        if (
            importlib.util.find_spec("concern") is None
            or importlib.util.find_spec("decoders") is None
        ):
            raise DependencyUnavailableError(
                f"{self.name} requires the official MhLiao/DB checkout at the revision in "
                "configs/models/dbnet.yaml on PYTHONPATH. See README 'DBNet/DBNet++ setup'."
            )
        if self.checkpoint is None or not self.checkpoint.is_file():
            expected = self.checkpoint or self.model_root / self.name / "model"
            raise CheckpointUnavailableError(
                f"{self.name} checkpoint not found at {expected}. Download the declared base "
                "checkpoint or provide a trained checkpoint; no fallback output is generated."
            )
        return self.checkpoint

    def detect(self, page: DocumentPage) -> list[DetectionRegion]:
        self._validate_runtime()
        raise DependencyUnavailableError(
            f"{self.name} runtime is revision-sensitive. Configure its official experiment "
            "YAML and prediction command in configs/models/dbnet.yaml. The adapter refuses "
            "to invoke an unverified API; use PaddleOCR detection until setup is completed."
        )
