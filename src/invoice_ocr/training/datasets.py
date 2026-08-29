"""Ground-truth validation and layout token/box alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from invoice_ocr.exceptions import AnnotationUnavailableError, InvalidGroundTruthError

STAGE_DIRECTORY = {
    "detector": "detection",
    "recognizer": "recognition",
    "layout": "layout",
}


@dataclass
class GroundTruthReport:
    root: Path
    final_count: int = 0
    detection_count: int = 0
    recognition_count: int = 0
    layout_count: int = 0
    table_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "valid": self.is_valid,
            "counts": {
                "final": self.final_count,
                "detection": self.detection_count,
                "recognition": self.recognition_count,
                "layout": self.layout_count,
                "tables": self.table_count,
            },
            "errors": self.errors,
        }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidGroundTruthError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InvalidGroundTruthError(f"ground-truth root must be an object: {path}")
    return payload


def _validate_detection(payload: dict[str, Any], path: Path) -> None:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise InvalidGroundTruthError(f"detection annotation needs non-empty 'pages': {path}")
    if not all(isinstance(page.get("regions"), list) for page in pages if isinstance(page, dict)):
        raise InvalidGroundTruthError(f"detection pages need 'regions' arrays: {path}")


def _validate_recognition(payload: dict[str, Any], path: Path) -> None:
    regions = payload.get("regions")
    if not isinstance(regions, list) or not regions:
        raise InvalidGroundTruthError(
            f"recognition annotation needs non-empty 'regions' with transcriptions: {path}"
        )
    if not all(
        isinstance(region, dict) and isinstance(region.get("text"), str) for region in regions
    ):
        raise InvalidGroundTruthError(f"every recognition region needs string 'text': {path}")


def _validate_layout(payload: dict[str, Any], path: Path) -> None:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise InvalidGroundTruthError(f"layout annotation needs non-empty 'pages': {path}")
    for page in pages:
        if not isinstance(page, dict):
            raise InvalidGroundTruthError(f"layout page must be an object: {path}")
        tokens, boxes, labels = page.get("tokens"), page.get("boxes"), page.get("labels")
        if (
            not isinstance(tokens, list)
            or not isinstance(boxes, list)
            or not isinstance(labels, list)
        ):
            raise InvalidGroundTruthError(
                f"layout page needs token, normalized box, and BIO label arrays: {path}"
            )
        if not tokens or len(tokens) != len(boxes) or len(tokens) != len(labels):
            raise InvalidGroundTruthError(
                f"layout tokens, boxes, and labels must be equal non-zero length: {path}"
            )
        ignore_mask = page.get("ignore_mask")
        if ignore_mask is not None and (
            not isinstance(ignore_mask, list)
            or len(ignore_mask) != len(tokens)
            or not all(isinstance(value, bool) for value in ignore_mask)
        ):
            raise InvalidGroundTruthError(
                f"layout ignore_mask must be a boolean array matching tokens: {path}"
            )
        for box in boxes:
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(value, int) and 0 <= value <= 1000 for value in box)
            ):
                raise InvalidGroundTruthError(
                    f"layout boxes must be four integers in the 0..1000 range: {path}"
                )


def validate_ground_truth(root: Path) -> GroundTruthReport:
    root = root.expanduser().resolve()
    report = GroundTruthReport(root=root)
    if not root.exists():
        report.errors.append(f"ground-truth directory does not exist: {root}")
        return report
    validators = {
        "detection": _validate_detection,
        "recognition": _validate_recognition,
        "layout": _validate_layout,
    }
    count_attributes = {
        "final": "final_count",
        "detection": "detection_count",
        "recognition": "recognition_count",
        "layout": "layout_count",
        "tables": "table_count",
    }
    for directory, attribute in count_attributes.items():
        paths = sorted((root / directory).rglob("*.json")) if (root / directory).is_dir() else []
        valid_count = 0
        for path in paths:
            try:
                payload = _load_json_object(path)
                validator = validators.get(directory)
                if validator is not None:
                    validator(payload, path)
                valid_count += 1
            except InvalidGroundTruthError as exc:
                report.errors.append(str(exc))
        setattr(report, attribute, valid_count)
    return report


def ensure_stage_annotations(root: Path, stage: str) -> list[Path]:
    if stage not in STAGE_DIRECTORY:
        raise ValueError(f"unknown training stage: {stage}")
    report = validate_ground_truth(root)
    if not report.is_valid:
        raise InvalidGroundTruthError("; ".join(report.errors))
    directory = STAGE_DIRECTORY[stage]
    paths = sorted((root / directory).glob("*.json"))
    if paths:
        return paths
    requirement = {
        "detector": ("detector boxes/polygons in GT/detection/<document_id>.json"),
        "recognizer": ("region transcriptions in GT/recognition/<document_id>.json"),
        "layout": (
            "OCR tokens, normalized token boxes, and BIO labels in GT/layout/<document_id>.json"
        ),
    }[stage]
    raise AnnotationUnavailableError(
        f"cannot train {stage}: no {requirement}. Final JSON is not converted into "
        "stage-level annotations."
    )


class WordIdEncoding(Protocol):
    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        """Return original word index for each encoded token."""


def align_word_labels(
    encoding: WordIdEncoding,
    word_labels: list[int],
    label_all_subtokens: bool = False,
) -> list[int]:
    """Align word-level BIO label IDs to wordpieces; special tokens use -100."""
    aligned: list[int] = []
    previous_word: int | None = None
    for word_id in encoding.word_ids():
        if word_id is None:
            aligned.append(-100)
        elif word_id < 0 or word_id >= len(word_labels):
            raise ValueError(f"tokenizer returned invalid word index: {word_id}")
        elif word_id != previous_word:
            aligned.append(word_labels[word_id])
        else:
            aligned.append(word_labels[word_id] if label_all_subtokens else -100)
        previous_word = word_id
    return aligned


def load_layout_pages(paths: list[Path]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_json_object(path)
        document_id = str(payload.get("document_id", path.stem))
        for page in payload["pages"]:
            pages.append(
                {
                    "document_id": document_id,
                    "page_index": int(page.get("page_index", len(pages))),
                    "image_path": page.get("image_path"),
                    "tokens": page["tokens"],
                    "boxes": page["boxes"],
                    "labels": page["labels"],
                    "ignore_mask": page.get("ignore_mask", [False] * len(page["tokens"])),
                }
            )
    return pages
