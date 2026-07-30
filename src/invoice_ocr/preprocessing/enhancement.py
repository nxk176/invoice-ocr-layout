"""Dependency-light document image enhancement."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance


def enhance_contrast(source: Path, destination: Path, factor: float = 1.2) -> None:
    if factor <= 0:
        raise ValueError("contrast factor must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        enhanced = ImageEnhance.Contrast(image.convert("RGB")).enhance(factor)
        enhanced.save(destination)

