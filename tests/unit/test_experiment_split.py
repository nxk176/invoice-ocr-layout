from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from invoice_ocr.experiments.evaluate_model import partition_annotation_paths
from invoice_ocr.experiments.split import (
    assert_locked_dataset_matches,
    create_locked_split,
    load_locked_split,
)
from invoice_ocr.io.paths import discover_documents


def _make_documents(root: Path, count: int = 10) -> list[str]:
    root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (16, 16), color=(index, index, index)).save(
            root / f"synthetic-{index}.png"
        )
    return [document.document_id for document in discover_documents(root)]


def test_locked_split_is_disjoint_complete_and_stable(tmp_path: Path) -> None:
    data = tmp_path / "data"
    gt = tmp_path / "GT"
    gt.mkdir()
    document_ids = _make_documents(data)
    output = gt / "splits" / "split_v1.json"
    created = create_locked_split(data, gt, output, seed=42)
    loaded = load_locked_split(output)

    train = set(loaded.train_document_ids)
    validation = set(loaded.validation_document_ids)
    test = set(loaded.test_document_ids)
    assert created.split_manifest_hash == loaded.split_manifest_hash
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(document_ids)
    assert loaded.random_seed == 42
    assert loaded.grouping_rules["pages_stay_together"] is True

    # The split file itself is excluded from the GT content hash.
    assert_locked_dataset_matches(loaded, data, gt)


def test_training_annotation_partition_never_contains_test_ids(tmp_path: Path) -> None:
    gt = tmp_path / "GT"
    directory = gt / "recognition"
    directory.mkdir(parents=True)
    train_ids = {"train-document"}
    validation_ids = {"validation-document"}
    test_ids = {"test-document"}
    for document_id in train_ids | validation_ids | test_ids:
        (directory / f"{document_id}.json").write_text(
            json.dumps(
                {
                    "document_id": document_id,
                    "regions": [
                        {
                            "page_index": 0,
                            "bbox": [0, 0, 10, 10],
                            "text": "synthetic",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    train, validation = partition_annotation_paths(
        gt,
        "recognizer",
        train_ids,
        validation_ids,
        test_ids,
    )
    assert [path.stem for path in train] == ["train-document"]
    assert [path.stem for path in validation] == ["validation-document"]
    assert all(path.stem not in test_ids for path in train + validation)
