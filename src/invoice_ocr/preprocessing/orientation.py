"""EXIF-based lossless orientation correction."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from invoice_ocr.contracts import OrientationMetadata


def correct_exif_orientation(source: Path, destination: Path) -> OrientationMetadata:
    """Apply explicit EXIF transpose and record the original orientation decision."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        exif_orientation = int(image.getexif().get(274, 1))
        rotation_by_exif = {1: 0, 3: 180, 6: 270, 8: 90}
        rotation = rotation_by_exif.get(exif_orientation, 0)
        corrected = ImageOps.exif_transpose(image).convert("RGB")
        corrected.save(destination)
    return OrientationMetadata(
        rotation_degrees=rotation,
        confidence=1.0 if exif_orientation in rotation_by_exif else None,
        method="exif" if exif_orientation in rotation_by_exif else "none",
    )
