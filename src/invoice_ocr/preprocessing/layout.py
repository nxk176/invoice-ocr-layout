"""Canonical OCR-region input contract for layout models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from invoice_ocr.contracts import BoundingBox


class LayoutTextRegion(Protocol):
    """Minimum region fields needed to build deterministic layout input."""

    @property
    def page_index(self) -> int: ...

    @property
    def region_id(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def bbox(self) -> BoundingBox: ...


RegionT = TypeVar("RegionT", bound=LayoutTextRegion)


def prepare_layout_regions(regions: Sequence[RegionT]) -> list[RegionT]:
    """Match pseudo-GT token filtering and deterministic geometric ordering."""
    return sorted(
        (region for region in regions if region.text.strip()),
        key=lambda region: (
            region.page_index,
            region.bbox.y_min,
            region.bbox.x_min,
            region.region_id,
        ),
    )
