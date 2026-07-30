"""Deterministic document-level train/validation/test splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "validation", "test"]


def split_for_document(
    document_id: str,
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> SplitName:
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("split ratios must reserve positive train and test partitions")
    digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    if unit < train_ratio:
        return "train"
    if unit < train_ratio + validation_ratio:
        return "validation"
    return "test"


def create_split_manifest(
    document_ids: list[str],
    destination: Path,
    seed: int,
    resume: bool = False,
) -> dict[str, object]:
    if destination.is_file() and resume:
        loaded = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or loaded.get("seed") != seed:
            raise ValueError("existing split manifest is invalid or uses a different seed")
        return loaded
    assignments = {
        document_id: split_for_document(document_id, seed)
        for document_id in sorted(set(document_ids))
    }
    manifest: dict[str, object] = {
        "seed": seed,
        "strategy": "sha256_document_id",
        "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "assignments": assignments,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
