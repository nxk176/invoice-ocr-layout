from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from invoice_ocr.experiments.evaluate_model import ModelEvaluationRequest
from invoice_ocr.experiments.evaluate_pipeline import PipelineEvaluationRequest
from invoice_ocr.experiments.orchestrator import (
    ExperimentRequest,
    run_experiment,
)
from invoice_ocr.experiments.split import create_locked_split, load_locked_split
from invoice_ocr.experiments.train_model import LockedTrainingRequest
from invoice_ocr.pipeline import PipelineSelection


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class MockProtocolExecutor:
    def __init__(self) -> None:
        self.evaluation_calls = 0
        self.pipeline_evaluation_calls = 0
        self.training_calls = 0

    def evaluate(self, request: ModelEvaluationRequest) -> Path:
        manifest_path = request.output_dir / "manifest.json"
        if request.resume and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                return request.output_dir
        self.evaluation_calls += 1
        split = load_locked_split(request.split_manifest)
        fine_tuned = request.checkpoint_source == "finetuned"
        if request.stage == "detector":
            metric_name = "detection_f1"
            value = 0.75 if fine_tuned else 0.50
            lower = False
        elif request.stage == "recognizer":
            metric_name = "recognition_cer"
            value = 0.20 if fine_tuned else 0.40
            lower = True
        else:
            metric_name = "entity_f1"
            value = 0.65 if fine_tuned else 0.35
            lower = False
        request.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            manifest_path,
            {
                "experiment_id": request.output_dir.name,
                "run_kind": "finetuned" if fine_tuned else "pretrained",
                "stage": request.stage,
                "model": request.model,
                "pipeline": {
                    "detector": None,
                    "recognizer": None,
                    "layout": None,
                },
                "baseline_mode": request.baseline_mode,
                "finetuned_mode": request.finetuned_mode,
                "checkpoint": {
                    "identifier": "synthetic",
                    "revision": "fixed-revision",
                    "sha256": "synthetic-hash",
                    "path": str(request.checkpoint) if request.checkpoint else None,
                    "best_epoch": 2 if fine_tuned else 1,
                },
                "split_manifest_hash": split.split_manifest_hash,
                "test_document_ids": split.test_document_ids,
                "schema_version": "schema-v1",
                "preprocessing_config_hash": "pre-v1",
                "postprocessing_config_hash": "post-v1",
                "workflow_defaults_hash": "workflow-v1",
                "validation_tolerance": request.validation_tolerance,
                "batch_size": request.batch_size,
                "num_workers": request.num_workers,
                "device": "cpu",
                "hardware_fingerprint": "synthetic-cpu",
                "metric_code_version": "metrics-v1",
                "status": "success",
                "skip_reason": None,
            },
        )
        _write_json(
            request.output_dir / "metrics.json",
            {
                "metrics": {
                    metric_name: {
                        "value": value,
                        "numerator": value * len(split.test_document_ids),
                        "denominator": len(split.test_document_ids),
                        "evaluated_sample_count": len(split.test_document_ids),
                        "skipped_sample_count": 0,
                        "lower_is_better": lower,
                        "na_reason": None,
                    }
                },
                "primary_metric": metric_name,
                "per_document": {
                    document_id: {metric_name: value} for document_id in split.test_document_ids
                },
                "per_field": {"invoice_number": value},
            },
        )
        _write_json(
            request.output_dir / "timing.json",
            {
                "total_wall_time_seconds": 1.5 if fine_tuned else 2.0,
                "model_load_time_seconds": 0.1,
                "preprocessing_time_seconds": 0.1,
                "detection_time_seconds": (0.5 if request.stage == "detector" else None),
                "recognition_time_seconds": (0.5 if request.stage == "recognizer" else None),
                "layout_inference_time_seconds": (0.5 if request.stage == "layout" else None),
                "processed_document_count": len(split.test_document_ids),
                "processed_page_count": len(split.test_document_ids),
                "failed_document_count": 0,
                "skipped_document_count": 0,
                "peak_cpu_ram_mb": 64.0,
                "peak_gpu_memory_mb": None,
                "warmup_iterations": request.warmup_iterations,
                "unavailable_reasons": {"peak_gpu_memory_mb": "synthetic CPU"},
            },
        )
        (request.output_dir / "errors.jsonl").write_text("", encoding="utf-8")
        predictions = request.output_dir / "predictions"
        predictions.mkdir()
        (predictions / "synthetic.jsonl").write_text(
            json.dumps({"synthetic": True}) + "\n", encoding="utf-8"
        )
        return request.output_dir

    def evaluate_pipeline(self, request: PipelineEvaluationRequest) -> Path:
        manifest_path = request.output_dir / "manifest.json"
        if request.resume and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                return request.output_dir
        self.pipeline_evaluation_calls += 1
        split = load_locked_split(request.split_manifest)
        fine_tuned = request.run_kind == "finetuned"
        value = 0.70 if fine_tuned else 0.40
        request.output_dir.mkdir(parents=True, exist_ok=True)
        fairness = {
            "split_manifest_hash": split.split_manifest_hash,
            "test_document_ids": split.test_document_ids,
            "schema_version": "schema-v1",
            "preprocessing_config_hash": "pre-v1",
            "postprocessing_config_hash": "post-v1",
            "workflow_defaults_hash": "workflow-v1",
            "validation_tolerance": request.validation_tolerance,
            "batch_size": request.batch_size,
            "num_workers": request.num_workers,
            "device": "cpu",
            "hardware_fingerprint": "synthetic-cpu",
            "metric_code_version": "metrics-v1",
        }
        _write_json(
            manifest_path,
            {
                "experiment_id": request.output_dir.name,
                "run_kind": request.run_kind,
                "stage": None,
                "model": None,
                "pipeline": {
                    "detector": request.pipeline.detector,
                    "recognizer": request.pipeline.recognizer,
                    "layout": request.pipeline.layout,
                },
                "checkpoint": {
                    "identifier": "synthetic-pipeline",
                    "revision": "fixed-revision",
                    "sha256": "synthetic-hash",
                },
                "status": "success",
                "skip_reason": None,
                **fairness,
            },
        )
        metric_names = (
            "final_field_exact_match",
            "final_normalized_field_accuracy",
            "final_medicine_row_matching",
            "final_document_exact_match",
            "schema_validation_success_rate",
        )
        _write_json(
            request.output_dir / "metrics.json",
            {
                "metrics": {
                    name: {
                        "value": value,
                        "numerator": value * len(split.test_document_ids),
                        "denominator": len(split.test_document_ids),
                        "evaluated_sample_count": len(split.test_document_ids),
                        "skipped_sample_count": 0,
                        "lower_is_better": False,
                        "na_reason": None,
                    }
                    for name in metric_names
                },
                "primary_metric": "final_normalized_field_accuracy",
                "per_document": {
                    document_id: {"final_normalized_field_accuracy": value}
                    for document_id in split.test_document_ids
                },
                "per_field": {"invoice_number": value},
            },
        )
        _write_json(
            request.output_dir / "timing.json",
            {
                "total_wall_time_seconds": 2.0 if fine_tuned else 3.0,
                "detection_time_seconds": 0.4,
                "recognition_time_seconds": 0.4,
                "layout_inference_time_seconds": 0.4,
                "processed_document_count": len(split.test_document_ids),
                "processed_page_count": len(split.test_document_ids),
                "failed_document_count": 0,
                "skipped_document_count": 0,
                "peak_cpu_ram_mb": 64.0,
                "peak_gpu_memory_mb": None,
                "warmup_iterations": request.warmup_iterations,
                "unavailable_reasons": {"peak_gpu_memory_mb": "synthetic CPU"},
            },
        )
        (request.output_dir / "errors.jsonl").write_text("", encoding="utf-8")
        predictions = request.output_dir / "predictions"
        predictions.mkdir(exist_ok=True)
        _write_json(predictions / "synthetic.json", {"synthetic": True})
        return request.output_dir

    def train(self, request: LockedTrainingRequest) -> Path:
        manifest_path = request.output_dir / "manifest.json"
        if request.resume and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                return request.output_dir
        self.training_calls += 1
        split = load_locked_split(request.split_manifest)
        best = request.output_dir / "best"
        best.mkdir(parents=True, exist_ok=True)
        _write_json(best / "synthetic_model.json", {"not_model_weights": True})
        _write_json(
            manifest_path,
            {
                "status": "success",
                "stage": request.stage,
                "model": request.model,
                "split_manifest_hash": split.split_manifest_hash,
                "train_document_ids": split.train_document_ids,
                "validation_document_ids": split.validation_document_ids,
                "test_document_ids": split.test_document_ids,
                "test_set_used_for_training": False,
                "skip_reason": None,
            },
        )
        _write_json(
            request.output_dir / "timing.json",
            {
                "total_training_time_seconds": 99.0,
                "training_samples": len(split.train_document_ids),
                "validation_samples": len(split.validation_document_ids),
            },
        )
        _write_json(
            request.output_dir / "training_metrics.json",
            {
                "status": "success",
                "selection_split": "validation",
                "best_validation_metric": 0.8,
            },
        )
        _write_json(
            request.output_dir / "checkpoints_manifest.json",
            {
                "status": "success",
                "selection_split": "validation",
                "best_checkpoint": {
                    "path": str(best),
                    "best_epoch": 2,
                    "best_validation_metric": 0.8,
                },
            },
        )
        return request.output_dir


def test_mock_pretrained_train_finetuned_compare_and_resume(tmp_path: Path) -> None:
    data = tmp_path / "data"
    gt = tmp_path / "GT"
    data.mkdir()
    gt.mkdir()
    for index in range(12):
        Image.new("RGB", (12, 12), color=(index, index, index)).save(
            data / f"synthetic-{index}.png"
        )
    split_path = gt / "splits" / "split_v1.json"
    split = create_locked_split(data, gt, split_path, seed=42)
    output = tmp_path / "outputs" / "experiment"
    executor = MockProtocolExecutor()
    request = ExperimentRequest(
        output_dir=output,
        data_root=data,
        gt_root=gt,
        split_manifest=split_path,
        pipeline=PipelineSelection("paddleocr", "vietocr", "layoutlmv3"),
        model_root=tmp_path / "models",
        work_root=tmp_path / "work",
        device="cpu",
        resume=True,
    )
    outcome = run_experiment(request, executor=executor)
    assert outcome.successful_count == 1
    assert executor.training_calls == 4  # linear probe plus three full fine-tunes
    assert executor.evaluation_calls == 6
    assert executor.pipeline_evaluation_calls == 2

    comparison = json.loads((output / "comparison" / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["same_test_split"] is True
    assert comparison["test_document_count"] == len(split.test_document_ids)
    assert comparison["metrics_after"]["entity_f1"] > comparison["metrics_before"]["entity_f1"]
    before_timing = json.loads((output / "pretrained" / "timing.json").read_text(encoding="utf-8"))
    after_timing = json.loads((output / "finetuned" / "timing.json").read_text(encoding="utf-8"))
    training_timing = json.loads((output / "training" / "timing.json").read_text(encoding="utf-8"))
    assert before_timing["total_wall_time_seconds"] == 3.0
    assert after_timing["total_wall_time_seconds"] == 2.0
    assert training_timing["total_training_time_seconds"] == 297.0
    assert after_timing["total_wall_time_seconds"] < training_timing["total_training_time_seconds"]
    first_counts = (
        executor.training_calls,
        executor.evaluation_calls,
        executor.pipeline_evaluation_calls,
    )
    run_experiment(request, executor=executor)
    assert (
        executor.training_calls,
        executor.evaluation_calls,
        executor.pipeline_evaluation_calls,
    ) == first_counts
