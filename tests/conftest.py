"""Shared synthetic test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def synthetic_image(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic" / "invoice.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (400, 300), color="white").save(path)
    return path
