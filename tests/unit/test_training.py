from __future__ import annotations

import json
from pathlib import Path

import pytest

from invoice_ocr.exceptions import AnnotationUnavailableError
from invoice_ocr.training.datasets import (
    align_word_labels,
    ensure_stage_annotations,
    load_layout_pages,
    validate_ground_truth,
)
from invoice_ocr.training.splits import create_split_manifest, split_for_document


def test_empty_gt_is_valid_but_has_no_annotations(tmp_path: Path) -> None:
    report = validate_ground_truth(tmp_path)
    assert report.is_valid
    assert report.final_count == 0
    with pytest.raises(AnnotationUnavailableError, match="boxes/polygons"):
        ensure_stage_annotations(tmp_path, "detector")
    with pytest.raises(AnnotationUnavailableError, match="transcriptions"):
        ensure_stage_annotations(tmp_path, "recognizer")
    with pytest.raises(AnnotationUnavailableError, match="normalized token boxes"):
        ensure_stage_annotations(tmp_path, "layout")


def test_layout_validation_requires_equal_token_box_label_lengths(tmp_path: Path) -> None:
    path = tmp_path / "layout" / "document.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "document_id": "document",
                "pages": [{"tokens": ["A"], "boxes": [], "labels": ["O"]}],
            }
        ),
        encoding="utf-8",
    )
    report = validate_ground_truth(tmp_path)
    assert not report.is_valid
    assert "equal non-zero length" in report.errors[0]


def test_layout_validation_and_loader_preserve_optional_ignore_mask(tmp_path: Path) -> None:
    path = tmp_path / "layout" / "document.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "document_id": "document",
                "pages": [
                    {
                        "tokens": ["Safe", "Uncertain"],
                        "boxes": [[0, 0, 10, 10], [10, 0, 20, 10]],
                        "labels": ["O", "O"],
                        "ignore_mask": [False, True],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert validate_ground_truth(tmp_path).is_valid
    assert load_layout_pages([path])[0]["ignore_mask"] == [False, True]


def test_layout_validation_rejects_misaligned_ignore_mask(tmp_path: Path) -> None:
    path = tmp_path / "layout" / "document.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "document_id": "document",
                "pages": [
                    {
                        "tokens": ["Synthetic"],
                        "boxes": [[0, 0, 10, 10]],
                        "labels": ["O"],
                        "ignore_mask": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_ground_truth(tmp_path)
    assert not report.is_valid
    assert "ignore_mask" in report.errors[0]


def test_split_is_deterministic_and_resume_keeps_manifest(tmp_path: Path) -> None:
    assert split_for_document("doc-a", 42) == split_for_document("doc-a", 42)
    path = tmp_path / "split_manifest.json"
    first = create_split_manifest(["doc-b", "doc-a"], path, 42)
    second = create_split_manifest(["different"], path, 42, resume=True)
    assert first == second


class FakeEncoding:
    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        return [None, 0, 0, 1, None]


def test_tokenizer_word_alignment_masks_special_and_continuation_tokens() -> None:
    assert align_word_labels(FakeEncoding(), [3, 4]) == [-100, 3, -100, 4, -100]
