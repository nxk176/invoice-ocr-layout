from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest
from PIL import Image

from invoice_ocr.contracts import (
    DocumentPage,
    OrientationMetadata,
    ProcessingStatus,
    SourceDocument,
)
from invoice_ocr.exceptions import NoInputDocumentsError
from invoice_ocr.io.jsonl import read_jsonl, write_jsonl
from invoice_ocr.io.paths import (
    deterministic_document_id,
    discover_documents,
    prediction_relative_path,
)
from invoice_ocr.io.pdf_render import render_document


def page_record() -> DocumentPage:
    return DocumentPage(
        document_id="document-123",
        source_path="synthetic.png",
        page_index=0,
        model_name="test",
        model_revision="revision",
        processing_status=ProcessingStatus.SUCCESS,
        image_path="rendered.png",
        width=100,
        height=200,
        orientation=OrientationMetadata(),
    )


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [page_record()])
    assert list(read_jsonl(path, DocumentPage)) == [page_record()]


def test_portable_path_inputs_generate_stable_ids() -> None:
    digest = "a" * 64
    windows = deterministic_document_id(Path(PureWindowsPath("folder/invoice.pdf")), digest)
    linux = deterministic_document_id(Path(PurePosixPath("folder/invoice.pdf")), digest)
    if sys.platform == "win32":
        assert windows == linux
    assert prediction_relative_path("nested/invoice.pdf").as_posix().endswith("nested/invoice.json")


def test_same_filename_in_different_directories_has_different_id() -> None:
    digest = "b" * 64
    assert deterministic_document_id(Path("a/invoice.pdf"), digest) != (
        deterministic_document_id(Path("b/invoice.pdf"), digest)
    )


def test_empty_data_directory_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(NoInputDocumentsError, match="no input documents found"):
        discover_documents(tmp_path)


def test_image_render_has_zero_based_page_index(synthetic_image: Path, tmp_path: Path) -> None:
    document = discover_documents(synthetic_image)[0]
    pages = render_document(document, tmp_path / "render")
    assert [page.page_index for page in pages] == [0]


def test_pdf_render_indexes_pages_from_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeBitmap:
        def to_pil(self) -> Image.Image:
            return Image.new("RGB", (20, 10), "white")

    class FakePage:
        def render(self, scale: float) -> FakeBitmap:
            assert scale > 0
            return FakeBitmap()

    class FakePdf:
        def __init__(self, path: str) -> None:
            self.path = path

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=FakePdf))
    document = SourceDocument(
        document_id="document-pdf",
        source_path=str(tmp_path / "synthetic.pdf"),
        relative_path="synthetic.pdf",
        media_type="pdf",
        sha256="c" * 64,
    )
    pages = render_document(document, tmp_path / "rendered")
    assert [page.page_index for page in pages] == [0, 1]
