"""PDF and image rendering into stable zero-based preprocessed page records."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

from invoice_ocr.contracts import (
    DocumentPage,
    OrientationMetadata,
    ProcessingStatus,
    SourceDocument,
)
from invoice_ocr.exceptions import DependencyUnavailableError
from invoice_ocr.preprocessing.deskew import deskew_image


def _orientation_metadata(image: Image.Image) -> OrientationMetadata:
    exif_orientation = int(image.getexif().get(274, 1))
    rotation_by_exif = {1: 0, 3: 180, 6: 270, 8: 90}
    rotation = rotation_by_exif.get(exif_orientation, 0)
    return OrientationMetadata(
        rotation_degrees=rotation,
        confidence=1.0 if exif_orientation in rotation_by_exif else None,
        method="exif" if exif_orientation in rotation_by_exif else "none",
    )


def _save_preprocessed(image: Image.Image, target: Path) -> tuple[int, int, float]:
    """Apply dependency-light enhancement and optional conservative OpenCV deskew."""
    enhanced = ImageEnhance.Contrast(image.convert("RGB")).enhance(1.1)
    enhanced.save(target)
    deskew_angle = 0.0
    if (
        importlib.util.find_spec("cv2") is not None
        and importlib.util.find_spec("numpy") is not None
    ):
        deskewed = target.with_name(f"{target.stem}.deskew{target.suffix}")
        deskew_angle = deskew_image(target, deskewed)
        deskewed.replace(target)
    return enhanced.width, enhanced.height, deskew_angle


def render_document(
    document: SourceDocument, output_dir: Path, dpi: int = 200
) -> list[DocumentPage]:
    """Render and preprocess a source; intermediate page indexes are zero-based."""
    if dpi <= 0:
        raise ValueError("render DPI must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if document.media_type == "image":
        with Image.open(document.source_path) as image:
            orientation = _orientation_metadata(image)
            corrected = ImageOps.exif_transpose(image)
            target = output_dir / f"{document.document_id}-p0000.png"
            width, height, deskew_angle = _save_preprocessed(corrected, target)
        return [
            DocumentPage(
                document_id=document.document_id,
                source_path=document.source_path,
                page_index=0,
                model_name="pillow-render",
                model_revision=None,
                processing_status=ProcessingStatus.SUCCESS,
                image_path=str(target),
                width=width,
                height=height,
                orientation=orientation,
                deskew_angle_degrees=deskew_angle,
            )
        ]
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise DependencyUnavailableError(
            "PDF input requires pypdfium2. Install the 'pdf' project extra."
        ) from exc
    pdf = pdfium.PdfDocument(document.source_path)
    scale = dpi / 72
    pages: list[DocumentPage] = []
    try:
        for page_index in range(len(pdf)):
            bitmap = pdf[page_index].render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            target = output_dir / f"{document.document_id}-p{page_index:04d}.png"
            width, height, deskew_angle = _save_preprocessed(image, target)
            pages.append(
                DocumentPage(
                    document_id=document.document_id,
                    source_path=document.source_path,
                    page_index=page_index,
                    model_name="pypdfium2",
                    model_revision=None,
                    processing_status=ProcessingStatus.SUCCESS,
                    image_path=str(target),
                    width=width,
                    height=height,
                    orientation=OrientationMetadata(method="pdf_render"),
                    deskew_angle_degrees=deskew_angle,
                )
            )
    finally:
        pdf.close()
    return pages
