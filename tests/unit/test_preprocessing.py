from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from invoice_ocr.io.paths import discover_documents
from invoice_ocr.io.pdf_render import render_document
from invoice_ocr.preprocessing.layout import prepare_layout_regions


def test_exif_orientation_is_applied_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    source = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (20, 10), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    document = discover_documents(source)[0]
    page = render_document(document, tmp_path / "rendered")[0]
    assert page.orientation.rotation_degrees == 270
    assert page.orientation.method == "exif"
    assert (page.width, page.height) == (10, 20)


def test_layout_regions_match_pseudo_gt_filtering_and_order() -> None:
    def region(region_id: str, text: str, y_min: float, x_min: float) -> SimpleNamespace:
        return SimpleNamespace(
            page_index=0,
            region_id=region_id,
            text=text,
            bbox=SimpleNamespace(y_min=y_min, x_min=x_min),
        )

    prepared = prepare_layout_regions(
        [
            region("bottom", "Bottom", 20, 5),
            region("blank", "  ", 0, 0),
            region("right", "Right", 10, 20),
            region("left", "Left", 10, 10),
        ]
    )

    assert [item.region_id for item in prepared] == ["left", "right", "bottom"]
