"""Deterministic geometric reading order."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from invoice_ocr.contracts import DetectionRegion, RecognizedRegion

Region = TypeVar("Region", DetectionRegion, RecognizedRegion)


def sort_reading_order(regions: Sequence[Region], line_tolerance: float = 0.5) -> list[Region]:
    """Sort top-to-bottom and left-to-right with height-aware line grouping."""
    if line_tolerance < 0:
        raise ValueError("line_tolerance must be non-negative")
    ordered = sorted(regions, key=lambda region: (region.bbox.y_min, region.bbox.x_min))
    lines: list[list[Region]] = []
    for region in ordered:
        region_height = region.bbox.y_max - region.bbox.y_min
        center_y = (region.bbox.y_min + region.bbox.y_max) / 2
        if not lines:
            lines.append([region])
            continue
        last = lines[-1]
        last_center = sum((item.bbox.y_min + item.bbox.y_max) / 2 for item in last) / len(last)
        if abs(center_y - last_center) <= max(1.0, region_height * line_tolerance):
            last.append(region)
        else:
            lines.append([region])
    return [region for line in lines for region in sorted(line, key=lambda item: item.bbox.x_min)]
