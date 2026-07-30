"""VI-LayoutXLM KIE adapter through official PaddleOCR/PP-Structure code."""

from __future__ import annotations

import importlib.util

from invoice_ocr.adapters.layout.base import LayoutAdapter
from invoice_ocr.contracts import DocumentPage, LabeledEntity, RecognizedRegion, Relation
from invoice_ocr.exceptions import CheckpointUnavailableError, DependencyUnavailableError


class VILayoutXLMAdapter(LayoutAdapter):
    name = "vi_layoutxlm"

    def _validate_runtime(self) -> None:
        if (
            importlib.util.find_spec("paddle") is None
            or importlib.util.find_spec("paddlenlp") is None
        ):
            raise DependencyUnavailableError(
                "VI-LayoutXLM requires compatible paddlepaddle and paddlenlp packages plus "
                "the official PaddleOCR PP-Structure KIE configuration. See README."
            )
        checkpoint = self.checkpoint or self.model_root / "vi_layoutxlm" / "invoice-best"
        if not checkpoint.is_dir():
            raise CheckpointUnavailableError(
                f"fine-tuned VI-LayoutXLM checkpoint not found at {checkpoint}. The base "
                "model does not know this repository's invoice label set."
            )

    def prepare(self) -> None:
        self._validate_runtime()

    def extract(
        self, page: DocumentPage, regions: list[RecognizedRegion]
    ) -> tuple[list[LabeledEntity], list[Relation]]:
        self._validate_runtime()
        raise DependencyUnavailableError(
            "VI-LayoutXLM inference must be launched with the revision-pinned official "
            "PP-Structure KIE config declared in configs/models/vi_layoutxlm.yaml. No sample "
            "entities are returned when that runtime is unavailable."
        )
