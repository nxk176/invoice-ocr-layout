from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from invoice_ocr.exceptions import AnnotationUnavailableError
from invoice_ocr.experiments.evaluate_model import (
    ModelEvaluationRequest,
    _layout_pretrained_guard,
    evaluate_model,
)
from invoice_ocr.experiments.split import create_locked_split
from invoice_ocr.experiments.train_model import LockedTrainingRequest, train_model_locked


def test_base_layout_model_with_random_head_is_not_reported_as_pretrained() -> None:
    request = ModelEvaluationRequest(
        stage="layout",
        model="layoutlmv3",
        checkpoint_source="pretrained",
        data_root=Path("data"),
        gt_root=Path("GT"),
        split_manifest=Path("GT/splits/split_v1.json"),
        output_dir=Path("outputs/synthetic"),
    )
    with pytest.raises(AnnotationUnavailableError, match="random invoice task head"):
        _layout_pretrained_guard(request)


def test_completed_evaluation_resume_returns_without_loading_data(tmp_path: Path) -> None:
    output = tmp_path / "completed"
    output.mkdir()
    for name, payload in (
        ("manifest.json", {"status": "success"}),
        ("metrics.json", {"metrics": {}}),
        ("timing.json", {"total_wall_time_seconds": 1.0}),
    ):
        (output / name).write_text(json.dumps(payload), encoding="utf-8")
    request = ModelEvaluationRequest(
        stage="recognizer",
        model="vietocr",
        checkpoint_source="pretrained",
        data_root=tmp_path / "missing-data",
        gt_root=tmp_path / "missing-gt",
        split_manifest=tmp_path / "missing-split.json",
        output_dir=output,
        resume=True,
    )
    assert evaluate_model(request) == output


def test_missing_locked_stage_annotations_create_skipped_not_fake_result(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    gt = tmp_path / "GT"
    data.mkdir()
    gt.mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8), color=(index, index, index)).save(data / f"synthetic-{index}.png")
    split = gt / "splits" / "split_v1.json"
    create_locked_split(data, gt, split, seed=42)
    output = tmp_path / "training"
    train_model_locked(
        LockedTrainingRequest(
            stage="recognizer",
            model="vietocr",
            checkpoint_source="pretrained",
            data_root=data,
            gt_root=gt,
            split_manifest=split,
            output_dir=output,
        )
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "training_metrics.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "skipped"
    assert manifest["test_set_used_for_training"] is False
    assert "GT/recognition" in manifest["skip_reason"]
    assert metrics["history"] == []
