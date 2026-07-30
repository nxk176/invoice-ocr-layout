"""Creation and validation of immutable document-level split manifests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invoice_ocr.experiments.contracts import LockedSplitManifest
from invoice_ocr.experiments.hashing import canonical_json_hash, directory_manifest_hash
from invoice_ocr.io.paths import discover_documents


def _partition_document_ids(
    document_ids: list[str], seed: int
) -> tuple[list[str], list[str], list[str]]:
    ordered = sorted(
        set(document_ids),
        key=lambda document_id: canonical_json_hash({"seed": seed, "document_id": document_id}),
    )
    count = len(ordered)
    if count == 0:
        return [], [], []
    if count == 1:
        return ordered, [], []
    if count == 2:
        return [ordered[0]], [], [ordered[1]]
    test_count = max(1, round(count * 0.1))
    validation_count = max(1, round(count * 0.1))
    if test_count + validation_count >= count:
        test_count = 1
        validation_count = 1
    train_end = count - validation_count - test_count
    validation_end = count - test_count
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]


def _hashable_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "split_manifest_hash"}


def create_locked_split(
    data_root: Path,
    gt_root: Path,
    output_path: Path,
    seed: int,
    force: bool = False,
) -> LockedSplitManifest:
    if output_path.exists() and not force:
        existing = load_locked_split(output_path)
        if existing.random_seed != seed:
            raise ValueError(
                f"locked split already exists with seed {existing.random_seed}; use --force "
                "only when intentionally creating a new protocol"
            )
        return existing
    documents = discover_documents(data_root)
    train_ids, validation_ids, test_ids = _partition_document_ids(
        [document.document_id for document in documents], seed
    )
    dataset_documents = [
        {
            "document_id": document.document_id,
            "relative_path": document.relative_path,
            "sha256": document.sha256,
        }
        for document in documents
    ]
    payload: dict[str, Any] = {
        "train_document_ids": train_ids,
        "validation_document_ids": validation_ids,
        "test_document_ids": test_ids,
        "random_seed": seed,
        "dataset_manifest_hash": canonical_json_hash(dataset_documents),
        "gt_manifest_hash": directory_manifest_hash(gt_root, ".json", exclude_prefixes=("splits",)),
        "split_manifest_hash": "0" * 64,
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "grouping_rules": {
            "unit": "source_document",
            "pages_stay_together": True,
            "strategy": "sha256_seeded_order",
            "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        },
        "dataset_documents": dataset_documents,
    }
    normalized = LockedSplitManifest.model_validate(payload).model_dump(mode="json")
    payload["split_manifest_hash"] = canonical_json_hash(_hashable_manifest(normalized))
    manifest = LockedSplitManifest.model_validate(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_locked_split(path: Path) -> LockedSplitManifest:
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}")
    manifest = LockedSplitManifest.model_validate_json(path.read_text(encoding="utf-8"))
    payload = manifest.model_dump(mode="json")
    expected = canonical_json_hash(_hashable_manifest(payload))
    if manifest.split_manifest_hash != expected:
        raise ValueError(
            f"split manifest hash mismatch at {path}: expected {expected}, "
            f"found {manifest.split_manifest_hash}"
        )
    return manifest


def assert_locked_dataset_matches(
    manifest: LockedSplitManifest, data_root: Path, gt_root: Path
) -> None:
    documents = discover_documents(data_root)
    current_dataset = [
        {
            "document_id": document.document_id,
            "relative_path": document.relative_path,
            "sha256": document.sha256,
        }
        for document in documents
    ]
    current_dataset_hash = canonical_json_hash(current_dataset)
    if current_dataset_hash != manifest.dataset_manifest_hash:
        raise ValueError(
            "dataset manifest hash differs from the locked split; create a new split version "
            "instead of silently changing test documents"
        )
    current_gt_hash = directory_manifest_hash(gt_root, ".json", exclude_prefixes=("splits",))
    if current_gt_hash != manifest.gt_manifest_hash:
        raise ValueError(
            "GT manifest hash differs from the locked split; create a new split version "
            "or restore the original annotations"
        )
