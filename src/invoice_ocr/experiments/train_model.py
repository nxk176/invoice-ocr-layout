"""Leakage-safe training initialized from declared pretrained checkpoints."""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from invoice_ocr.exceptions import (
    AnnotationUnavailableError,
    CheckpointUnavailableError,
    ConfigurationError,
    DependencyUnavailableError,
)
from invoice_ocr.experiments.contracts import TrainingTiming
from invoice_ocr.experiments.evaluate_model import Stage, partition_annotation_paths
from invoice_ocr.experiments.hashing import (
    canonical_json_hash,
    directory_manifest_hash,
    reproducibility_metadata,
)
from invoice_ocr.experiments.runtime import PeakCpuMonitor, TorchCudaHooks
from invoice_ocr.experiments.split import assert_locked_dataset_matches, load_locked_split
from invoice_ocr.experiments.training_modes import (
    BestCheckpointSelector,
    CheckpointCandidate,
    LayoutTrainingMode,
    configure_layout_trainability,
)
from invoice_ocr.model_manifest import load_adapter_manifest
from invoice_ocr.pipeline import resolve_device
from invoice_ocr.training.datasets import load_layout_pages
from invoice_ocr.training.detector import train_detector
from invoice_ocr.training.layout import (
    LayoutPageDataset,
    layout_trainer_progress_arguments,
    load_bio_label_mapping,
)
from invoice_ocr.training.recognizer import train_recognizer


@dataclass
class LockedTrainingRequest:
    stage: Stage
    model: str
    checkpoint_source: str
    data_root: Path
    gt_root: Path
    split_manifest: Path
    output_dir: Path
    layout_gt_root: Path | None = None
    model_root: Path = Path("models")
    device: str = "auto"
    seed: int = 42
    resume: bool = False
    force: bool = False
    layout_training_mode: LayoutTrainingMode = "full_finetune"
    selection_metric: str | None = None
    maximize_metric: bool | None = None
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision_mode: str = "none"
    optimizer: str = "adamw_torch"
    scheduler: str = "linear"
    num_workers: int = 0


@dataclass(frozen=True)
class TrainingBackendResult:
    candidates: list[CheckpointCandidate]
    epochs_completed: float
    time_per_epoch_seconds: float | None
    early_stopping_epoch: float | None
    log_history: list[dict[str, Any]]
    training_samples: int
    validation_samples: int


class TrainingBackend(Protocol):
    def train(
        self,
        request: LockedTrainingRequest,
        train_annotations: list[Path],
        validation_annotations: list[Path],
        device: str,
    ) -> TrainingBackendResult: ...


def default_selection_metric(stage: Stage) -> tuple[str, bool]:
    return {
        "detector": ("detection_hmean", True),
        "recognizer": ("recognition_cer", False),
        "layout": ("entity_f1", True),
    }[stage]


def _write_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _copy_best_checkpoint(source: Path, destination: Path, output_root: Path) -> None:
    resolved_destination = destination.resolve()
    resolved_output = output_root.resolve()
    if resolved_destination.parent != resolved_output or resolved_destination.name != "best":
        raise ValueError(
            f"unsafe best-checkpoint destination outside training output: {destination}"
        )
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    elif source.is_file():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)
    else:
        raise CheckpointUnavailableError(f"validation-selected checkpoint does not exist: {source}")


def _classification_metrics(eval_prediction: Any) -> dict[str, float]:
    import numpy as np

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    valid = labels != -100
    expected = labels[valid]
    predicted = predictions[valid]
    if expected.size == 0:
        return {"entity_f1": 0.0, "token_accuracy": 0.0}
    correct = int((predicted == expected).sum())
    true_positive = int(((predicted == expected) & (expected != 0)).sum())
    false_positive = int(((predicted != expected) & (predicted != 0)).sum())
    false_negative = int(((predicted != expected) & (expected != 0)).sum())
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    entity_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "entity_f1": entity_f1,
        "token_accuracy": correct / int(expected.size),
    }


def _layout_base_checkpoint(request: LockedTrainingRequest) -> Path:
    name = "layoutlmv3-base" if request.model == "layoutlmv3" else "vi-layoutxlm-base"
    checkpoint = request.model_root / name
    if not checkpoint.is_dir():
        raise CheckpointUnavailableError(
            f"pretrained encoder checkpoint not found at {checkpoint}; "
            f"run python scripts/download_models.py --model {name}"
        )
    return checkpoint


class ProductionTrainingBackend:
    """Dispatch official backends and require validation evidence for best selection."""

    def train(
        self,
        request: LockedTrainingRequest,
        train_annotations: list[Path],
        validation_annotations: list[Path],
        device: str,
    ) -> TrainingBackendResult:
        if request.stage == "layout":
            return self._train_layout(request, train_annotations, validation_annotations, device)
        started = time.perf_counter()
        backend_output = request.output_dir / "backend"
        if request.stage == "detector":
            train_detector(
                request.model,
                train_annotations,
                request.data_root,
                backend_output,
                request.resume,
                device,
            )
        else:
            train_recognizer(
                request.model,
                train_annotations,
                request.data_root,
                backend_output,
                request.resume,
                device,
            )
        metrics_file = backend_output / "validation_metrics.jsonl"
        if not metrics_file.is_file():
            raise ConfigurationError(
                f"official {request.model} training did not emit "
                f"{metrics_file}. Configure validation on locked validation IDs and emit "
                "JSON lines containing checkpoint, epoch, metric, and split='validation'; "
                "the framework will not select a checkpoint from training or test metrics."
            )
        candidates: list[CheckpointCandidate] = []
        log_history: list[dict[str, Any]] = []
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"invalid validation metric record in {metrics_file}")
            candidate = CheckpointCandidate(
                path=Path(str(raw["checkpoint"])),
                epoch=int(raw["epoch"]),
                metric_value=float(raw["metric"]),
                evaluated_split=str(raw["split"]),
            )
            candidates.append(candidate)
            log_history.append(raw)
        elapsed = time.perf_counter() - started
        epochs = max((candidate.epoch for candidate in candidates), default=0)
        return TrainingBackendResult(
            candidates=candidates,
            epochs_completed=float(epochs),
            time_per_epoch_seconds=elapsed / epochs if epochs else None,
            early_stopping_epoch=None,
            log_history=log_history,
            training_samples=len(train_annotations),
            validation_samples=len(validation_annotations),
        )

    def _train_layout(
        self,
        request: LockedTrainingRequest,
        train_annotations: list[Path],
        validation_annotations: list[Path],
        device: str,
    ) -> TrainingBackendResult:
        if request.model == "vi_layoutxlm":
            raise DependencyUnavailableError(
                "VI-LayoutXLM locked training requires the revision-pinned official "
                "PaddleOCR PP-Structure KIE command to emit validation_metrics.jsonl. "
                "No Transformers substitute or random-head baseline is used."
            )
        if request.model != "layoutlmv3":
            raise ConfigurationError(f"unsupported layout model: {request.model}")
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoProcessor,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise DependencyUnavailableError(
                "LayoutLMv3 training requires torch and transformers==4.44.2"
            ) from exc
        base_checkpoint = _layout_base_checkpoint(request)
        labels, label_to_id = load_bio_label_mapping(Path("configs/labels/invoice_bio_labels.yaml"))
        processor = AutoProcessor.from_pretrained(base_checkpoint, apply_ocr=False)
        model = AutoModelForTokenClassification.from_pretrained(
            base_checkpoint,
            num_labels=len(labels),
            id2label={index: label for index, label in enumerate(labels)},
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
        )
        configure_layout_trainability(model, request.layout_training_mode)
        parameter_rows = list(model.named_parameters())
        train_pages = load_layout_pages(train_annotations)
        validation_pages = load_layout_pages(validation_annotations)
        image_root = request.layout_gt_root or request.data_root
        train_dataset = LayoutPageDataset(train_pages, processor, label_to_id, image_root)
        validation_dataset = LayoutPageDataset(validation_pages, processor, label_to_id, image_root)
        backend_output = request.output_dir / "backend"
        metric_name, maximize = _selection_settings(request)
        fp16 = request.mixed_precision_mode == "fp16"
        bf16 = request.mixed_precision_mode == "bf16"
        arguments = TrainingArguments(
            output_dir=str(backend_output),
            seed=request.seed,
            data_seed=request.seed,
            num_train_epochs=request.epochs,
            learning_rate=request.learning_rate,
            per_device_train_batch_size=request.batch_size,
            per_device_eval_batch_size=request.batch_size,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            dataloader_num_workers=request.num_workers,
            optim=request.optimizer,
            lr_scheduler_type=request.scheduler,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            greater_is_better=maximize,
            save_total_limit=3,
            report_to=[],
            remove_unused_columns=False,
            fp16=fp16,
            bf16=bf16,
            **layout_trainer_progress_arguments(
                len(train_dataset),
                request.batch_size,
                request.gradient_accumulation_steps,
            ),
            use_cpu=device == "cpu",
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=_classification_metrics,
        )
        result = trainer.train(resume_from_checkpoint=request.resume)
        best_source = (
            Path(str(trainer.state.best_model_checkpoint))
            if trainer.state.best_model_checkpoint
            else backend_output
        )
        if best_source == backend_output:
            trainer.save_model(best_source)
        # Trainer checkpoints do not persist AutoProcessor unless explicitly attached.
        # End-to-end LayoutLM inference needs the exact processor beside the selected model.
        processor.save_pretrained(best_source)
        best_metric = trainer.state.best_metric
        if best_metric is None or not math.isfinite(float(best_metric)):
            raise RuntimeError(
                "Trainer did not produce a finite validation metric; no best checkpoint selected"
            )
        validation_key = f"eval_{metric_name.removeprefix('eval_')}"
        best_epochs = [
            math.ceil(float(entry["epoch"]))
            for entry in trainer.state.log_history
            if (
                "epoch" in entry
                and isinstance(entry.get(validation_key), (int, float))
                and math.isclose(
                    float(entry[validation_key]),
                    float(best_metric),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            )
        ]
        best_epoch = (
            min(best_epochs)
            if best_epochs
            else math.ceil(float(trainer.state.epoch or request.epochs))
        )
        candidates = [
            CheckpointCandidate(
                path=best_source,
                epoch=best_epoch,
                metric_value=float(best_metric),
                evaluated_split="validation",
            )
        ]
        log_history = [dict(entry) for entry in trainer.state.log_history]
        log_history.append(
            {
                "layout_training_mode": request.layout_training_mode,
                "trainable_parameter_count": sum(
                    parameter.numel() for _, parameter in parameter_rows if parameter.requires_grad
                ),
                "frozen_parameter_count": sum(
                    parameter.numel()
                    for _, parameter in parameter_rows
                    if not parameter.requires_grad
                ),
                "selection_split": "validation",
            }
        )
        epochs_completed = float(trainer.state.epoch or request.epochs)
        training_runtime = float(result.metrics.get("train_runtime", 0.0))
        return TrainingBackendResult(
            candidates=candidates,
            epochs_completed=epochs_completed,
            time_per_epoch_seconds=(
                training_runtime / epochs_completed if epochs_completed else None
            ),
            early_stopping_epoch=None,
            log_history=log_history,
            training_samples=len(train_pages),
            validation_samples=len(validation_pages),
        )


def _selection_settings(request: LockedTrainingRequest) -> tuple[str, bool]:
    default_metric, default_maximize = default_selection_metric(request.stage)
    return (
        request.selection_metric or default_metric,
        request.maximize_metric if request.maximize_metric is not None else default_maximize,
    )


def _training_config(
    request: LockedTrainingRequest,
    train_ids: list[str],
    validation_ids: list[str],
    test_ids: list[str],
) -> dict[str, Any]:
    return {
        "stage": request.stage,
        "model": request.model,
        "checkpoint_source": request.checkpoint_source,
        "layout_training_mode": (
            request.layout_training_mode if request.stage == "layout" else None
        ),
        "layout_gt_root": (
            str(request.layout_gt_root)
            if request.stage == "layout" and request.layout_gt_root is not None
            else None
        ),
        "train_document_ids": train_ids,
        "validation_document_ids": validation_ids,
        "excluded_test_document_ids": test_ids,
        "epochs": request.epochs,
        "learning_rate": request.learning_rate,
        "batch_size": request.batch_size,
        "gradient_accumulation_steps": request.gradient_accumulation_steps,
        "mixed_precision_mode": request.mixed_precision_mode,
        "optimizer": request.optimizer,
        "scheduler": request.scheduler,
        "seed": request.seed,
    }


def _write_skipped_training(
    request: LockedTrainingRequest,
    reason: str,
    split_hash: str,
    test_ids: list[str],
) -> Path:
    reproducibility = reproducibility_metadata()
    skipped_at = datetime.now(timezone.utc)
    timing = TrainingTiming(
        total_training_time_seconds=0.0,
        total_epochs_completed=0.0,
        time_per_epoch_seconds=None,
        best_epoch=None,
        best_validation_metric=None,
        early_stopping_epoch=None,
        training_samples=0,
        validation_samples=0,
        optimizer=request.optimizer,
        learning_rate=request.learning_rate,
        scheduler=request.scheduler,
        effective_batch_size=request.batch_size * request.gradient_accumulation_steps,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        mixed_precision_mode=request.mixed_precision_mode,
        peak_cpu_ram_mb=None,
        peak_gpu_memory_mb=None,
        unavailable_reasons={
            "peak_cpu_ram_mb": "training did not start",
            "peak_gpu_memory_mb": "training did not start",
        },
    )
    _write_object(request.output_dir / "timing.json", timing.model_dump(mode="json"))
    _write_object(
        request.output_dir / "training_metrics.json",
        {"status": "SKIPPED", "reason": reason, "history": []},
    )
    _write_object(
        request.output_dir / "checkpoints_manifest.json",
        {
            "status": "SKIPPED",
            "reason": reason,
            "selection_split": "validation",
            "checkpoints": [],
        },
    )
    _write_object(
        request.output_dir / "manifest.json",
        {
            "experiment_id": request.output_dir.name,
            "run_kind": "training",
            "stage": request.stage,
            "model": request.model,
            "status": "skipped",
            "skip_reason": reason,
            "split_manifest_hash": split_hash,
            "test_document_ids": test_ids,
            "test_set_used_for_training": False,
            "reproducibility": reproducibility,
            "command_line": sys.argv,
            "start_time": skipped_at.isoformat(),
            "end_time": skipped_at.isoformat(),
        },
    )
    return request.output_dir


def _resume_complete(output_dir: Path) -> bool:
    manifest = output_dir / "manifest.json"
    best = output_dir / "best"
    if not manifest.is_file() or not best.exists():
        return False
    return _load_manifest_status(manifest) == "success"


def _load_manifest_status(path: Path) -> str | None:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return str(loaded.get("status")) if isinstance(loaded, dict) else None


def train_model_locked(
    request: LockedTrainingRequest,
    backend: TrainingBackend | None = None,
) -> Path:
    """Train on train only, select on validation only, and never expose test to backend."""
    started_at = datetime.now(timezone.utc)
    if request.checkpoint_source != "pretrained":
        raise ConfigurationError(
            "locked fine-tuning must initialize from checkpoint-source=pretrained"
        )
    if request.epochs <= 0 or request.batch_size <= 0:
        raise ConfigurationError("epochs and batch size must be positive")
    if request.gradient_accumulation_steps <= 0:
        raise ConfigurationError("gradient accumulation steps must be positive")
    if request.output_dir.exists() and request.resume and _resume_complete(request.output_dir):
        return request.output_dir
    if (
        request.output_dir.exists()
        and not request.force
        and not request.resume
        and any(request.output_dir.iterdir())
    ):
        raise ConfigurationError(
            f"training output already exists: {request.output_dir}; use --resume or --force"
        )
    split = load_locked_split(request.split_manifest)
    assert_locked_dataset_matches(split, request.data_root, request.gt_root)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    config = _training_config(
        request,
        split.train_document_ids,
        split.validation_document_ids,
        split.test_document_ids,
    )
    (request.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if not split.train_document_ids:
        return _write_skipped_training(
            request,
            "locked split contains no train_document_ids",
            split.split_manifest_hash,
            split.test_document_ids,
        )
    if not split.validation_document_ids:
        return _write_skipped_training(
            request,
            "locked split contains no validation_document_ids; best checkpoint cannot be "
            "selected without validation data",
            split.split_manifest_hash,
            split.test_document_ids,
        )
    try:
        annotation_root = (
            request.layout_gt_root
            if request.stage == "layout" and request.layout_gt_root is not None
            else request.gt_root
        )
        train_annotations, validation_annotations = partition_annotation_paths(
            annotation_root,
            request.stage,
            set(split.train_document_ids),
            set(split.validation_document_ids),
            set(split.test_document_ids),
        )
    except AnnotationUnavailableError as exc:
        return _write_skipped_training(
            request, str(exc), split.split_manifest_hash, split.test_document_ids
        )
    missing_train = sorted(
        set(split.train_document_ids) - {path.stem for path in train_annotations}
    )
    missing_validation = sorted(
        set(split.validation_document_ids) - {path.stem for path in validation_annotations}
    )
    if missing_train or missing_validation:
        reason = (
            f"missing {request.stage} annotations under {annotation_root} for locked train IDs "
            f"{missing_train} "
            f"or validation IDs {missing_validation}; test annotations are never substituted"
        )
        return _write_skipped_training(
            request, reason, split.split_manifest_hash, split.test_document_ids
        )
    device = resolve_device(request.device)
    cuda = TorchCudaHooks()
    cuda.synchronize()
    cuda.reset_peak_memory_stats()
    cpu = PeakCpuMonitor()
    cpu.start()
    started = time.perf_counter()
    training_backend = backend or ProductionTrainingBackend()
    try:
        result = training_backend.train(request, train_annotations, validation_annotations, device)
        metric_name, maximize = _selection_settings(request)
        selector = BestCheckpointSelector(metric_name, greater_is_better=maximize)
        for candidate in result.candidates:
            selector.add(candidate)
        best = selector.best()
        if best.evaluated_split != "validation":
            raise RuntimeError("best checkpoint selection did not use validation")
        _copy_best_checkpoint(best.path, request.output_dir / "best", request.output_dir)
    except Exception as exc:
        cpu.stop()
        return _write_skipped_training(
            request, str(exc), split.split_manifest_hash, split.test_document_ids
        )
    finally:
        cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_cpu = cpu.stop()
    peak_gpu = cuda.peak_memory_mb()
    unavailable = {}
    if peak_gpu is None:
        unavailable["peak_gpu_memory_mb"] = cuda.unavailable_reason or "GPU peak memory unavailable"
    timing = TrainingTiming(
        total_training_time_seconds=elapsed,
        total_epochs_completed=result.epochs_completed,
        time_per_epoch_seconds=result.time_per_epoch_seconds,
        best_epoch=float(best.epoch),
        best_validation_metric=best.metric_value,
        early_stopping_epoch=result.early_stopping_epoch,
        training_samples=result.training_samples,
        validation_samples=result.validation_samples,
        optimizer=request.optimizer,
        learning_rate=request.learning_rate,
        scheduler=request.scheduler,
        effective_batch_size=request.batch_size * request.gradient_accumulation_steps,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        mixed_precision_mode=request.mixed_precision_mode,
        peak_cpu_ram_mb=peak_cpu,
        peak_gpu_memory_mb=peak_gpu,
        unavailable_reasons=unavailable,
    )
    _write_object(request.output_dir / "timing.json", timing.model_dump(mode="json"))
    _write_object(
        request.output_dir / "training_metrics.json",
        {
            "status": "success",
            "selection_metric": metric_name,
            "selection_split": "validation",
            "best_validation_metric": best.metric_value,
            "history": result.log_history,
        },
    )
    checkpoint_hash = directory_manifest_hash(request.output_dir / "best")
    checkpoint_manifest = {
        "status": "success",
        "selection_metric": metric_name,
        "maximize": maximize,
        "selection_split": "validation",
        "best_checkpoint": {
            "path": str(request.output_dir / "best"),
            "sha256": checkpoint_hash,
            "best_epoch": best.epoch,
            "best_validation_metric": best.metric_value,
        },
        "candidates": [
            {
                "path": str(candidate.path),
                "epoch": candidate.epoch,
                "metric": candidate.metric_value,
                "evaluated_split": candidate.evaluated_split,
            }
            for candidate in result.candidates
        ],
    }
    _write_object(request.output_dir / "checkpoints_manifest.json", checkpoint_manifest)
    _write_object(
        request.output_dir / "checkpoint_selection.json",
        {
            "best_epoch": best.epoch,
            "best_validation_metric": best.metric_value,
            "selection_metric": metric_name,
            "selection_split": "validation",
        },
    )
    source_manifest = load_adapter_manifest(request.stage, request.model)
    _write_object(
        request.output_dir / "manifest.json",
        {
            "experiment_id": request.output_dir.name,
            "run_kind": "training",
            "stage": request.stage,
            "model": request.model,
            "status": "success",
            "checkpoint_source": "pretrained",
            "pretrained_checkpoint": {
                "identifier": source_manifest.get("checkpoint_identifier"),
                "revision": source_manifest.get("checkpoint_revision")
                or source_manifest.get("revision"),
                "sha256": source_manifest.get("sha256"),
            },
            "finetuned_checkpoint": {
                "path": str(request.output_dir / "best"),
                "sha256": checkpoint_hash,
                "best_epoch": best.epoch,
            },
            "layout_training_mode": (
                request.layout_training_mode if request.stage == "layout" else None
            ),
            "split_manifest_hash": split.split_manifest_hash,
            "train_document_ids": split.train_document_ids,
            "validation_document_ids": split.validation_document_ids,
            "test_document_ids": split.test_document_ids,
            "test_set_used_for_training": False,
            "data_manifest_hash": split.dataset_manifest_hash,
            "gt_manifest_hash": split.gt_manifest_hash,
            "random_seed": request.seed,
            "training_hyperparameters": config,
            "reproducibility": reproducibility_metadata(),
            "command_line": sys.argv,
            "start_time": started_at.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "training_contract_hash": canonical_json_hash(config),
        },
    )
    return request.output_dir
