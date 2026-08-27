"""End-to-end pretrained-versus-fine-tuned experiment orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from invoice_ocr.experiments.comparison import compare_runs
from invoice_ocr.experiments.evaluate_model import (
    ModelEvaluationRequest,
    Stage,
    evaluate_model,
)
from invoice_ocr.experiments.evaluate_pipeline import (
    PipelineEvaluationRequest,
    evaluate_pipeline,
)
from invoice_ocr.experiments.hashing import canonical_json_hash
from invoice_ocr.experiments.split import (
    assert_locked_dataset_matches,
    load_locked_split,
)
from invoice_ocr.experiments.train_model import (
    LockedTrainingRequest,
    train_model_locked,
)
from invoice_ocr.experiments.training_modes import LayoutTrainingMode
from invoice_ocr.pipeline import PipelineSelection, enumerate_pipeline_combinations


@dataclass
class ExperimentRequest:
    output_dir: Path
    data_root: Path
    gt_root: Path
    split_manifest: Path
    layout_gt_root: Path | None = None
    gt_prefix: str | None = None
    target_manifest: Path | None = None
    pipeline: PipelineSelection | None = None
    all_combinations: bool = False
    protocol: str = "pretrained-vs-finetuned"
    layout_baseline_mode: LayoutTrainingMode = "linear_probe"
    layout_finetuned_mode: LayoutTrainingMode = "full_finetune"
    model_root: Path = Path("models")
    work_root: Path = Path("work")
    workflow_defaults: Path | None = None
    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 0
    warmup_iterations: int = 0
    seed: int = 42
    resume: bool = False
    force: bool = False
    fail_fast: bool = False
    epochs: int = 3
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    mixed_precision_mode: str = "none"
    validation_tolerance: str = "0.01"


@dataclass
class ExperimentOutcome:
    output_dir: Path
    combination_count: int
    successful_count: int
    skipped_count: int
    failed_count: int
    reasons: list[str] = field(default_factory=list)

    @property
    def all_failed(self) -> bool:
        return self.successful_count == 0


class ExperimentExecutor(Protocol):
    def evaluate(self, request: ModelEvaluationRequest) -> Path: ...

    def evaluate_pipeline(self, request: PipelineEvaluationRequest) -> Path: ...

    def train(self, request: LockedTrainingRequest) -> Path: ...


class ProductionExperimentExecutor:
    def evaluate(self, request: ModelEvaluationRequest) -> Path:
        return evaluate_model(request)

    def evaluate_pipeline(self, request: PipelineEvaluationRequest) -> Path:
        return evaluate_pipeline(request)

    def train(self, request: LockedTrainingRequest) -> Path:
        return train_model_locked(request)


def _load_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _write_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _pipeline_name(pipeline: PipelineSelection) -> str:
    return f"{pipeline.detector}__{pipeline.recognizer}__{pipeline.layout}"


def _models(pipeline: PipelineSelection) -> list[tuple[Stage, str]]:
    return [
        ("detector", pipeline.detector),
        ("recognizer", pipeline.recognizer),
        ("layout", pipeline.layout),
    ]


def _common_evaluation_request(
    request: ExperimentRequest,
    pipeline: PipelineSelection,
    stage: Stage,
    model: str,
    output_dir: Path,
    checkpoint_source: str,
    checkpoint: Path | None,
    baseline_mode: str | None,
    finetuned_mode: str | None,
) -> ModelEvaluationRequest:
    return ModelEvaluationRequest(
        stage=stage,
        model=model,
        checkpoint_source=checkpoint_source,
        checkpoint=checkpoint,
        layout_gt_root=request.layout_gt_root,
        data_root=request.data_root,
        gt_root=request.gt_root,
        split_manifest=request.split_manifest,
        split="test",
        output_dir=output_dir,
        model_root=request.model_root,
        work_root=request.work_root / _pipeline_name(pipeline),
        workflow_defaults=request.workflow_defaults,
        device=request.device,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        warmup_iterations=request.warmup_iterations,
        seed=request.seed,
        resume=request.resume,
        force=request.force,
        baseline_mode=baseline_mode,
        finetuned_mode=finetuned_mode,
        validation_tolerance=request.validation_tolerance,
    )


def _common_pipeline_evaluation_request(
    request: ExperimentRequest,
    pipeline: PipelineSelection,
    output_dir: Path,
    run_kind: str,
    checkpoints: dict[str, Path | None],
) -> PipelineEvaluationRequest:
    return PipelineEvaluationRequest(
        pipeline=pipeline,
        checkpoints=checkpoints,
        run_kind=run_kind,
        data_root=request.data_root,
        gt_root=request.gt_root,
        split_manifest=request.split_manifest,
        output_dir=output_dir,
        model_root=request.model_root,
        work_root=request.work_root / _pipeline_name(pipeline) / run_kind,
        workflow_defaults=request.workflow_defaults,
        device=request.device,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        warmup_iterations=request.warmup_iterations,
        seed=request.seed,
        resume=request.resume,
        force=request.force,
        validation_tolerance=request.validation_tolerance,
        baseline_mode=request.layout_baseline_mode if run_kind == "pretrained" else None,
        finetuned_mode=request.layout_finetuned_mode if run_kind == "finetuned" else None,
        gt_prefix=request.gt_prefix,
        target_manifest=request.target_manifest,
    )


def _common_training_request(
    request: ExperimentRequest,
    stage: Stage,
    model: str,
    output_dir: Path,
    layout_mode: LayoutTrainingMode,
) -> LockedTrainingRequest:
    return LockedTrainingRequest(
        stage=stage,
        model=model,
        checkpoint_source="pretrained",
        data_root=request.data_root,
        gt_root=request.gt_root,
        split_manifest=request.split_manifest,
        output_dir=output_dir,
        layout_gt_root=request.layout_gt_root,
        model_root=request.model_root,
        device=request.device,
        seed=request.seed,
        resume=request.resume,
        force=request.force,
        layout_training_mode=layout_mode,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        batch_size=request.batch_size,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        mixed_precision_mode=request.mixed_precision_mode,
        num_workers=request.num_workers,
    )


def _stage_status(path: Path) -> tuple[str, str | None]:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return "failed", f"missing manifest: {manifest}"
    payload = _load_object(manifest)
    return str(payload.get("status", "failed")), (
        str(payload["skip_reason"]) if payload.get("skip_reason") else None
    )


def _write_failed_stage(
    output_dir: Path,
    stage: Stage,
    model: str,
    reason: str,
    split_hash: str,
    test_ids: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_object(
        output_dir / "manifest.json",
        {
            "experiment_id": output_dir.name,
            "run_kind": "pretrained",
            "stage": stage,
            "model": model,
            "status": "skipped",
            "skip_reason": reason,
            "split_manifest_hash": split_hash,
            "test_document_ids": test_ids,
        },
    )
    _write_object(
        output_dir / "metrics.json",
        {"metrics": {}, "primary_metric": "", "per_document": {}, "per_field": {}},
    )
    _write_object(
        output_dir / "timing.json",
        {
            "total_wall_time_seconds": 0.0,
            "processed_document_count": 0,
            "processed_page_count": 0,
            "failed_document_count": 0,
            "skipped_document_count": len(test_ids),
            "unavailable_reasons": {"all": reason},
        },
    )
    (output_dir / "errors.jsonl").write_text(
        json.dumps({"status": "SKIPPED", "stage": stage, "reason": reason}) + "\n",
        encoding="utf-8",
    )


def _write_failed_pipeline(
    output_dir: Path,
    pipeline: PipelineSelection,
    reason: str,
    split_hash: str,
    test_ids: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = (
        "final_field_exact_match",
        "final_normalized_field_accuracy",
        "final_field_level_precision",
        "final_field_level_recall",
        "final_field_level_f1",
        "final_normalized_field_level_precision",
        "final_normalized_field_level_recall",
        "final_normalized_field_level_f1",
        "final_character_error_rate",
        "final_medicine_row_matching",
        "final_document_exact_match",
        "validation_success_rate",
        "unresolved_required_field_count",
        "schema_validation_success_rate",
    )
    metrics = {
        name: {
            "value": None,
            "numerator": None,
            "denominator": None,
            "evaluated_sample_count": 0,
            "skipped_sample_count": len(test_ids),
            "lower_is_better": name
            in {"final_character_error_rate", "unresolved_required_field_count"},
            "na_reason": reason,
        }
        for name in metric_names
    }
    _write_object(
        output_dir / "manifest.json",
        {
            "experiment_id": output_dir.parent.name,
            "stage": None,
            "model": None,
            "pipeline": {
                "detector": pipeline.detector,
                "recognizer": pipeline.recognizer,
                "layout": pipeline.layout,
            },
            "status": "skipped",
            "skip_reason": reason,
            "split_manifest_hash": split_hash,
            "test_document_ids": test_ids,
        },
    )
    _write_object(
        output_dir / "metrics.json",
        {
            "metrics": metrics,
            "primary_metric": "final_normalized_field_accuracy",
            "per_document": {},
            "per_field": {},
        },
    )
    _write_object(
        output_dir / "timing.json",
        {
            "total_wall_time_seconds": 0.0,
            "processed_document_count": 0,
            "processed_page_count": 0,
            "failed_document_count": 0,
            "skipped_document_count": len(test_ids),
            "unavailable_reasons": {"all": reason},
        },
    )
    (output_dir / "errors.jsonl").write_text(
        json.dumps({"status": "SKIPPED", "stage": "pipeline", "reason": reason}) + "\n",
        encoding="utf-8",
    )


def _aggregate_evaluations(
    root: Path,
    pipeline: PipelineSelection,
    kind: str,
) -> tuple[str, list[str]]:
    stage_dirs = [root / "stages" / stage for stage, _ in _models(pipeline)]
    end_to_end_dir = root / "end_to_end"
    evaluation_dirs = [*stage_dirs, end_to_end_dir]
    manifests = [
        _load_object(path / "manifest.json")
        for path in evaluation_dirs
        if (path / "manifest.json").is_file()
    ]
    metric_payloads = [
        _load_object(path / "metrics.json")
        for path in evaluation_dirs
        if (path / "metrics.json").is_file()
    ]
    timing_payloads = [
        _load_object(path / "timing.json")
        for path in evaluation_dirs
        if (path / "timing.json").is_file()
    ]
    reasons = [
        str(manifest["skip_reason"]) for manifest in manifests if manifest.get("skip_reason")
    ]
    merged_metrics: dict[str, Any] = {}
    per_document: dict[str, dict[str, float]] = {}
    per_field: dict[str, float] = {}
    primary_metrics: list[str] = []
    for payload in metric_payloads:
        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            merged_metrics.update(metrics)
        primary = payload.get("primary_metric")
        if primary:
            primary_metrics.append(str(primary))
        document_payload = payload.get("per_document", {})
        if isinstance(document_payload, dict):
            for document_id, values in document_payload.items():
                if isinstance(values, dict):
                    per_document.setdefault(str(document_id), {}).update(
                        {
                            str(name): float(value)
                            for name, value in values.items()
                            if isinstance(value, (int, float))
                        }
                    )
        field_payload = payload.get("per_field", {})
        if isinstance(field_payload, dict):
            per_field.update(
                {
                    str(name): float(value)
                    for name, value in field_payload.items()
                    if isinstance(value, (int, float))
                }
            )
    preferred_primary = {
        "finetuned": f"{pipeline.layout}_entity_f1",
        "pretrained": f"{pipeline.layout}_entity_f1",
    }.get(kind, "")
    primary_metric = (
        preferred_primary
        if preferred_primary in merged_metrics
        else (primary_metrics[-1] if primary_metrics else "")
    )
    _write_object(
        root / "metrics.json",
        {
            "metrics": merged_metrics,
            "primary_metric": primary_metric,
            "per_document": per_document,
            "per_field": per_field,
        },
    )
    time_fields = (
        "total_wall_time_seconds",
        "model_load_time_seconds",
        "preprocessing_time_seconds",
        "detection_time_seconds",
        "recognition_time_seconds",
        "layout_inference_time_seconds",
        "table_reconstruction_time_seconds",
        "postprocessing_time_seconds",
        "validation_time_seconds",
        "evaluation_time_seconds",
    )
    aggregate_timing: dict[str, Any] = {
        timing_field: sum(
            float(payload[timing_field])
            for payload in timing_payloads
            if isinstance(payload.get(timing_field), (int, float))
        )
        for timing_field in time_fields
    }
    for timing_field in (
        "processed_document_count",
        "processed_page_count",
        "failed_document_count",
        "skipped_document_count",
    ):
        values = [
            int(payload[timing_field])
            for payload in timing_payloads
            if isinstance(payload.get(timing_field), (int, float))
        ]
        aggregate_timing[timing_field] = max(values, default=0)
    for timing_field in ("peak_cpu_ram_mb", "peak_gpu_memory_mb"):
        values = [
            float(payload[timing_field])
            for payload in timing_payloads
            if isinstance(payload.get(timing_field), (int, float))
        ]
        aggregate_timing[timing_field] = max(values) if values else None
    aggregate_timing["warmup_iterations"] = max(
        (int(payload.get("warmup_iterations", 0)) for payload in timing_payloads),
        default=0,
    )
    aggregate_timing["unavailable_reasons"] = {
        f"stage_{index}": reason for index, reason in enumerate(reasons)
    }
    end_timing_path = end_to_end_dir / "timing.json"
    if end_timing_path.is_file():
        aggregate_timing = _load_object(end_timing_path)
    _write_object(root / "timing.json", aggregate_timing)
    reference = manifests[-1] if manifests else {}
    status = "success" if reference.get("status") == "success" else "skipped"
    top_manifest = {
        **reference,
        "experiment_id": root.parent.name,
        "run_kind": kind,
        "stage": None,
        "model": None,
        "pipeline": {
            "detector": pipeline.detector,
            "recognizer": pipeline.recognizer,
            "layout": pipeline.layout,
        },
        "checkpoint": {
            "identifier": "pipeline",
            "revision": canonical_json_hash(
                [manifest.get("checkpoint", {}) for manifest in manifests]
            ),
            "sha256": canonical_json_hash(
                [manifest.get("checkpoint", {}).get("sha256") for manifest in manifests]
            ),
        },
        "status": status,
        "skip_reason": "; ".join(reasons) if reasons else None,
    }
    _write_object(root / "manifest.json", top_manifest)
    errors: list[str] = []
    for path in evaluation_dirs:
        error_path = path / "errors.jsonl"
        if error_path.is_file():
            errors.extend(
                line for line in error_path.read_text(encoding="utf-8").splitlines() if line
            )
    (root / "errors.jsonl").write_text(
        "\n".join(errors) + ("\n" if errors else ""), encoding="utf-8"
    )
    end_predictions = end_to_end_dir / "predictions"
    if end_predictions.is_dir():
        shutil.copytree(end_predictions, root / "predictions", dirs_exist_ok=True)
    config = {
        "pipeline": top_manifest["pipeline"],
        "kind": kind,
        "locked_test_document_ids": top_manifest.get("test_document_ids", []),
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return status, reasons


def _aggregate_training(
    root: Path,
    pipeline: PipelineSelection,
    stage_dirs: list[Path],
) -> tuple[str, list[str]]:
    manifests = [
        _load_object(path / "manifest.json")
        for path in stage_dirs
        if (path / "manifest.json").is_file()
    ]
    timings = [
        _load_object(path / "timing.json")
        for path in stage_dirs
        if (path / "timing.json").is_file()
    ]
    checkpoint_manifests = [
        _load_object(path / "checkpoints_manifest.json")
        for path in stage_dirs
        if (path / "checkpoints_manifest.json").is_file()
    ]
    reasons = [
        str(manifest["skip_reason"]) for manifest in manifests if manifest.get("skip_reason")
    ]
    status = "success" if any(item.get("status") == "success" for item in manifests) else "skipped"
    total_training_time = sum(
        float(item.get("total_training_time_seconds", 0.0)) for item in timings
    )
    _write_object(
        root / "timing.json",
        {
            "total_training_time_seconds": total_training_time,
            "per_stage": timings,
        },
    )
    _write_object(
        root / "training_metrics.json",
        {
            "status": status,
            "selection_split": "validation",
            "stage_statuses": [
                {
                    "stage": manifest.get("stage"),
                    "model": manifest.get("model"),
                    "status": manifest.get("status"),
                    "reason": manifest.get("skip_reason"),
                }
                for manifest in manifests
            ],
        },
    )
    _write_object(
        root / "checkpoints_manifest.json",
        {
            "status": status,
            "selection_split": "validation",
            "stages": checkpoint_manifests,
        },
    )
    _write_object(
        root / "manifest.json",
        {
            "experiment_id": root.parent.name,
            "run_kind": "training",
            "pipeline": {
                "detector": pipeline.detector,
                "recognizer": pipeline.recognizer,
                "layout": pipeline.layout,
            },
            "status": status,
            "skip_reason": "; ".join(reasons) if reasons else None,
            "test_set_used_for_training": False,
        },
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "pipeline": {
                    "detector": pipeline.detector,
                    "recognizer": pipeline.recognizer,
                    "layout": pipeline.layout,
                },
                "selection_split": "validation",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return status, reasons


def _run_one(
    request: ExperimentRequest,
    pipeline: PipelineSelection,
    output_dir: Path,
    executor: ExperimentExecutor,
) -> tuple[str, list[str]]:
    split = load_locked_split(request.split_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(request.split_manifest, output_dir / "split_manifest.json")
    config = {
        "protocol": request.protocol,
        "pipeline": {
            "detector": pipeline.detector,
            "recognizer": pipeline.recognizer,
            "layout": pipeline.layout,
        },
        "layout_baseline_mode": request.layout_baseline_mode,
        "layout_finetuned_mode": request.layout_finetuned_mode,
        "split_manifest_hash": split.split_manifest_hash,
        "test_document_ids": split.test_document_ids,
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "device": request.device,
        "warmup_iterations": request.warmup_iterations,
        "validation_tolerance": request.validation_tolerance,
    }
    (output_dir / "experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    stage_reasons: list[str] = []
    pretrained_root = output_dir / "pretrained"
    training_root = output_dir / "training"
    finetuned_root = output_dir / "finetuned"
    training_stage_dirs: list[Path] = []
    baseline_by_stage: dict[str, Path | None] = {}
    baseline_ready: set[Stage] = set()
    best_by_stage: dict[Stage, Path] = {}
    finetuned_ready: set[Stage] = set()
    for stage, model in _models(pipeline):
        pretrained_stage = pretrained_root / "stages" / stage
        baseline_checkpoint: Path | None = None
        checkpoint_source = "pretrained"
        if stage == "layout":
            if request.layout_baseline_mode != "linear_probe":
                reason = (
                    "layout baseline must be linear_probe unless an official compatible "
                    "generic KIE checkpoint is explicitly configured"
                )
                _write_failed_stage(
                    pretrained_stage,
                    stage,
                    model,
                    reason,
                    split.split_manifest_hash,
                    split.test_document_ids,
                )
                stage_reasons.append(f"{stage}/{model}: {reason}")
            else:
                linear_training = pretrained_root / "linear_probe_training"
                try:
                    executor.train(
                        _common_training_request(
                            request, stage, model, linear_training, "linear_probe"
                        )
                    )
                    linear_status, linear_reason = _stage_status(linear_training)
                    if linear_status == "success":
                        baseline_checkpoint = linear_training / "best"
                        checkpoint_source = "linear_probe"
                    else:
                        reason = linear_reason or "linear probe training did not complete"
                        _write_failed_stage(
                            pretrained_stage,
                            stage,
                            model,
                            reason,
                            split.split_manifest_hash,
                            split.test_document_ids,
                        )
                        stage_reasons.append(f"{stage}/{model}: {reason}")
                except Exception as exc:
                    reason = str(exc)
                    _write_failed_stage(
                        pretrained_stage,
                        stage,
                        model,
                        reason,
                        split.split_manifest_hash,
                        split.test_document_ids,
                    )
                    stage_reasons.append(f"{stage}/{model}: {reason}")
        if not (pretrained_stage / "manifest.json").is_file():
            try:
                executor.evaluate(
                    _common_evaluation_request(
                        request,
                        pipeline,
                        stage,
                        model,
                        pretrained_stage,
                        checkpoint_source,
                        baseline_checkpoint,
                        request.layout_baseline_mode if stage == "layout" else "pretrained",
                        None,
                    )
                )
            except Exception as exc:
                reason = str(exc)
                _write_failed_stage(
                    pretrained_stage,
                    stage,
                    model,
                    reason,
                    split.split_manifest_hash,
                    split.test_document_ids,
                )
                stage_reasons.append(f"{stage}/{model}: {reason}")
        baseline_status, baseline_reason = _stage_status(pretrained_stage)
        if baseline_status == "success":
            baseline_ready.add(stage)
            baseline_by_stage[stage] = baseline_checkpoint
        elif baseline_reason:
            stage_reasons.append(f"{stage}/{model}: {baseline_reason}")
        training_stage = training_root / "stages" / stage
        training_stage_dirs.append(training_stage)
        try:
            executor.train(
                _common_training_request(
                    request,
                    stage,
                    model,
                    training_stage,
                    (request.layout_finetuned_mode if stage == "layout" else "full_finetune"),
                )
            )
            training_status, training_reason = _stage_status(training_stage)
            if training_status == "success":
                best_by_stage[stage] = training_stage / "best"
            elif training_reason:
                stage_reasons.append(f"{stage}/{model}: {training_reason}")
        except Exception as exc:
            reason = str(exc)
            stage_reasons.append(f"{stage}/{model}: {reason}")
        finetuned_stage = finetuned_root / "stages" / stage
        best = best_by_stage.get(stage)
        if best is None:
            _write_failed_stage(
                finetuned_stage,
                stage,
                model,
                "fine-tuned evaluation skipped because validation-selected best checkpoint "
                "is unavailable",
                split.split_manifest_hash,
                split.test_document_ids,
            )
            continue
        try:
            executor.evaluate(
                _common_evaluation_request(
                    request,
                    pipeline,
                    stage,
                    model,
                    finetuned_stage,
                    "finetuned",
                    best,
                    None,
                    request.layout_finetuned_mode if stage == "layout" else "full_finetune",
                )
            )
        except Exception as exc:
            reason = str(exc)
            _write_failed_stage(
                finetuned_stage,
                stage,
                model,
                reason,
                split.split_manifest_hash,
                split.test_document_ids,
            )
            stage_reasons.append(f"{stage}/{model}: {reason}")
        finetuned_status, finetuned_reason = _stage_status(finetuned_stage)
        if finetuned_status == "success":
            finetuned_ready.add(stage)
        elif finetuned_reason:
            stage_reasons.append(f"{stage}/{model}: {finetuned_reason}")
        if request.fail_fast and stage_reasons:
            break
    required_stages: set[Stage] = {"detector", "recognizer", "layout"}
    pretrained_end_to_end = pretrained_root / "end_to_end"
    if baseline_ready == required_stages:
        try:
            executor.evaluate_pipeline(
                _common_pipeline_evaluation_request(
                    request,
                    pipeline,
                    pretrained_end_to_end,
                    "pretrained",
                    baseline_by_stage,
                )
            )
        except Exception as exc:
            reason = str(exc)
            _write_failed_pipeline(
                pretrained_end_to_end,
                pipeline,
                reason,
                split.split_manifest_hash,
                split.test_document_ids,
            )
            stage_reasons.append(f"pipeline/pretrained: {reason}")
    else:
        missing = sorted(required_stages - baseline_ready)
        reason = f"pretrained end-to-end skipped because stages are unavailable: {missing}"
        _write_failed_pipeline(
            pretrained_end_to_end,
            pipeline,
            reason,
            split.split_manifest_hash,
            split.test_document_ids,
        )
        stage_reasons.append(reason)
    finetuned_end_to_end = finetuned_root / "end_to_end"
    if finetuned_ready == required_stages and set(best_by_stage) == required_stages:
        try:
            executor.evaluate_pipeline(
                _common_pipeline_evaluation_request(
                    request,
                    pipeline,
                    finetuned_end_to_end,
                    "finetuned",
                    {stage: path for stage, path in best_by_stage.items()},
                )
            )
        except Exception as exc:
            reason = str(exc)
            _write_failed_pipeline(
                finetuned_end_to_end,
                pipeline,
                reason,
                split.split_manifest_hash,
                split.test_document_ids,
            )
            stage_reasons.append(f"pipeline/finetuned: {reason}")
    else:
        missing = sorted(required_stages - finetuned_ready)
        reason = f"fine-tuned end-to-end skipped because stages are unavailable: {missing}"
        _write_failed_pipeline(
            finetuned_end_to_end,
            pipeline,
            reason,
            split.split_manifest_hash,
            split.test_document_ids,
        )
        stage_reasons.append(reason)
    pretrained_status, before_reasons = _aggregate_evaluations(
        pretrained_root, pipeline, "pretrained"
    )
    training_status, training_reasons = _aggregate_training(
        training_root, pipeline, training_stage_dirs
    )
    finetuned_status, after_reasons = _aggregate_evaluations(finetuned_root, pipeline, "finetuned")
    reasons = list(dict.fromkeys(stage_reasons + before_reasons + training_reasons + after_reasons))
    comparison_root = output_dir / "comparison"
    try:
        compare_runs(pretrained_root, finetuned_root, comparison_root)
    except (OSError, ValueError) as exc:
        reasons.append(f"comparison: {exc}")
    summary = comparison_root / "summary.md"
    if reasons:
        comparison_root.mkdir(parents=True, exist_ok=True)
        if not summary.is_file():
            summary.write_text("# Pretrained vs fine-tuned\n", encoding="utf-8")
        with summary.open("a", encoding="utf-8") as stream:
            stream.write("\n## SKIPPED/failed models\n\n")
            for reason in reasons:
                stream.write(f"- {reason}\n")
    success = (
        pretrained_status == "success"
        and training_status == "success"
        and finetuned_status == "success"
        and (comparison_root / "comparison.json").is_file()
    )
    status = "success" if success else "skipped"
    _write_object(
        output_dir / "manifest.json",
        {
            "experiment_id": output_dir.name,
            "protocol": request.protocol,
            "pipeline": config["pipeline"],
            "split_manifest_hash": split.split_manifest_hash,
            "test_document_ids": split.test_document_ids,
            "status": status,
            "reasons": reasons,
            "test_set_used_for_training": False,
            "comparison_created": (comparison_root / "comparison.json").is_file(),
        },
    )
    return status, reasons


def run_experiment(
    request: ExperimentRequest,
    executor: ExperimentExecutor | None = None,
) -> ExperimentOutcome:
    """Run one or all 12 combinations while isolating model-specific failures."""
    if request.protocol != "pretrained-vs-finetuned":
        raise ValueError(f"unsupported experiment protocol: {request.protocol}")
    if request.all_combinations == (request.pipeline is not None):
        raise ValueError("select exactly one of --pipeline or --all-combinations")
    split = load_locked_split(request.split_manifest)
    assert_locked_dataset_matches(split, request.data_root, request.gt_root)
    pipelines = (
        enumerate_pipeline_combinations() if request.all_combinations else [request.pipeline]
    )
    selected = [pipeline for pipeline in pipelines if pipeline is not None]
    runner = executor or ProductionExperimentExecutor()
    successful = skipped = failed = 0
    all_reasons: list[str] = []
    for pipeline in selected:
        output_dir = (
            request.output_dir / _pipeline_name(pipeline)
            if request.all_combinations
            else request.output_dir
        )
        try:
            status, reasons = _run_one(request, pipeline, output_dir, runner)
            all_reasons.extend(f"{_pipeline_name(pipeline)}: {reason}" for reason in reasons)
            if status == "success":
                successful += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            all_reasons.append(f"{_pipeline_name(pipeline)}: {exc}")
            if request.fail_fast:
                break
    outcome = ExperimentOutcome(
        output_dir=request.output_dir,
        combination_count=len(selected),
        successful_count=successful,
        skipped_count=skipped,
        failed_count=failed,
        reasons=all_reasons,
    )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    _write_object(
        request.output_dir / "experiment_summary.json",
        {
            "protocol": request.protocol,
            "combination_count": outcome.combination_count,
            "successful_count": outcome.successful_count,
            "skipped_count": outcome.skipped_count,
            "failed_count": outcome.failed_count,
            "all_failed": outcome.all_failed,
            "reasons": outcome.reasons,
        },
    )
    return outcome
