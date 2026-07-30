"""UTF-8 JSON Lines serialization with validated Pydantic records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path, records: Iterable[BaseModel], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(record.model_dump_json(exclude_none=False))
            stream.write("\n")


def read_jsonl(path: Path, model_type: type[T]) -> Iterator[T]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield model_type.model_validate_json(line)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc


def completed_document_ids(path: Path, model_type: type[T]) -> set[str]:
    """Return document IDs with at least one successful persisted record."""
    return {
        record.document_id
        for record in read_jsonl(path, model_type)
        if getattr(record, "processing_status", None) == "success"
    }
