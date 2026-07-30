"""PDF and image rendering into stable zero-based page records."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from invoice_ocr.contracts import (
    DocumentPage,
    OrientationMetadata,
    ProcessingStatus,
    SourceDocument,
)
from invoice_ocr.exceptions import DependencyUnavailableError


def render_document(document: SourceDocument, output_dir: Path, dpi: int = 200) -> list[DocumentPage]:
    """Render a source document; intermediate page indexes are always zero-based."""
    if dpi <= 0:
        raise ValueError("render DPI must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if document.media_type == "image":
        with Image.open(document.source_path) as image:
            rgb = image.convert("RGB")
            target = output_dir / f"{document.document_id}-p0000.png"
            rgb.save(target)
            return [
                DocumentPage(
                    document_id=document.document_id,
                    source_path=document.source_path,
                    page_index=0,
                    model_name="pillow-render",
                    model_revision=None,
                    processing_status=ProcessingStatus.SUCCESS,
                    image_path=str(target),
                    width=rgb.width,
                    height=rgb.height,
                    orientation=OrientationMetadata(),
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
            image.save(target)
            pages.append(
                DocumentPage(
                    document_id=document.document_id,
                    source_path=document.source_path,
                    page_index=page_index,
                    model_name="pypdfium2",
                    model_revision=None,
                    processing_status=ProcessingStatus.SUCCESS,
                    image_path=str(target),
                    width=image.width,
                    height=image.height,
                    orientation=OrientationMetadata(),
                )
            )
    finally:
        pdf.close()
    return pages

