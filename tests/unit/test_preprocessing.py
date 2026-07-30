from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from invoice_ocr.io.paths import discover_documents
from invoice_ocr.io.pdf_render import render_document


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
