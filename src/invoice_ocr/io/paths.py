"""Portable document discovery and deterministic identifier helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from invoice_ocr.contracts import SourceDocument
from invoice_ocr.exceptions import NoInputDocumentsError

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_document_id(relative_path: Path, content_sha256: str) -> str:
    """Create a stable ID that disambiguates equal filenames in different folders."""
    portable = relative_path.as_posix().casefold()
    payload = f"{portable}\0{content_sha256}".encode()
    return hashlib.sha256(payload).hexdigest()


def discover_documents(input_path: Path) -> list[SourceDocument]:
    """Discover supported files without following hidden generated directories."""
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise NoInputDocumentsError(f"input path does not exist: {input_path}")
    candidates = (
        [input_path]
        if input_path.is_file()
        else sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
        )
    )
    candidates = [path for path in candidates if path.suffix.casefold() in SUPPORTED_SUFFIXES]
    if not candidates:
        raise NoInputDocumentsError(f"no input documents found in {input_path}")
    root = input_path.parent if input_path.is_file() else input_path
    documents: list[SourceDocument] = []
    for candidate in candidates:
        relative = candidate.relative_to(root)
        digest = sha256_file(candidate)
        documents.append(
            SourceDocument(
                document_id=deterministic_document_id(relative, digest),
                source_path=str(candidate),
                relative_path=relative.as_posix(),
                media_type="pdf" if candidate.suffix.casefold() == ".pdf" else "image",
                sha256=digest,
            )
        )
    return documents


def prediction_relative_path(relative_source_path: str) -> Path:
    """Preserve directory structure while replacing the source extension with JSON."""
    return Path(relative_source_path).with_suffix(".json")

