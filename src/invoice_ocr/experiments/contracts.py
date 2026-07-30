"""Validated contracts for reproducible experiments, timings, and metrics."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LockedSplitManifest(ExperimentModel):
    train_document_ids: list[str]
    validation_document_ids: list[str]
    test_document_ids: list[str]
    random_seed: int
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gt_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    creation_time: datetime
    grouping_rules: dict[str, Any]
    dataset_documents: list[dict[str, str]]

    @model_validator(mode="after")
    def disjoint_partitions(self) -> LockedSplitManifest:
        train = set(self.train_document_ids)
        validation = set(self.validation_document_ids)
        test = set(self.test_document_ids)
        if train & validation or train & test or validation & test:
            raise ValueError("train, validation, and test document IDs must not overlap")
        known = {str(item["document_id"]) for item in self.dataset_documents}
        assigned = train | validation | test
        if assigned != known:
            missing = sorted(known - assigned)
            unknown = sorted(assigned - known)
            raise ValueError(
                "split assignment must match dataset documents; "
                f"missing={missing}, unknown={unknown}"
            )
        return self

    def ids_for(self, split: Literal["train", "validation", "test"]) -> list[str]:
        return {
            "train": self.train_document_ids,
            "validation": self.validation_document_ids,
            "test": self.test_document_ids,
        }[split]


class AggregateMetric(ExperimentModel):
    value: float | None
    numerator: float | int | None
    denominator: float | int | None
    evaluated_sample_count: int = Field(ge=0)
    skipped_sample_count: int = Field(ge=0)
    lower_is_better: bool
    na_reason: str | None = None

    @model_validator(mode="after")
    def require_na_reason(self) -> AggregateMetric:
        if self.value is None and not self.na_reason:
            raise ValueError("an unavailable metric must include na_reason")
        return self


class EvaluationTiming(ExperimentModel):
    total_wall_time_seconds: float = Field(ge=0)
    model_load_time_seconds: float | None = Field(default=None, ge=0)
    preprocessing_time_seconds: float | None = Field(default=None, ge=0)
    detection_time_seconds: float | None = Field(default=None, ge=0)
    recognition_time_seconds: float | None = Field(default=None, ge=0)
    layout_inference_time_seconds: float | None = Field(default=None, ge=0)
    table_reconstruction_time_seconds: float | None = Field(default=None, ge=0)
    postprocessing_time_seconds: float | None = Field(default=None, ge=0)
    validation_time_seconds: float | None = Field(default=None, ge=0)
    evaluation_time_seconds: float | None = Field(default=None, ge=0)
    mean_time_per_document_seconds: float | None = Field(default=None, ge=0)
    median_time_per_document_seconds: float | None = Field(default=None, ge=0)
    mean_time_per_page_seconds: float | None = Field(default=None, ge=0)
    throughput_documents_per_second: float | None = Field(default=None, ge=0)
    throughput_pages_per_second: float | None = Field(default=None, ge=0)
    peak_cpu_ram_mb: float | None = Field(default=None, ge=0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0)
    processed_document_count: int = Field(ge=0)
    processed_page_count: int = Field(ge=0)
    failed_document_count: int = Field(ge=0)
    skipped_document_count: int = Field(ge=0)
    warmup_iterations: int = Field(ge=0)
    unavailable_reasons: dict[str, str] = Field(default_factory=dict)


class TrainingTiming(ExperimentModel):
    total_training_time_seconds: float = Field(ge=0)
    total_epochs_completed: float = Field(ge=0)
    time_per_epoch_seconds: float | None = Field(default=None, ge=0)
    best_epoch: float | None = Field(default=None, ge=0)
    best_validation_metric: float | None = None
    early_stopping_epoch: float | None = Field(default=None, ge=0)
    training_samples: int = Field(ge=0)
    validation_samples: int = Field(ge=0)
    optimizer: str
    learning_rate: float = Field(gt=0)
    scheduler: str
    effective_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    mixed_precision_mode: str
    peak_cpu_ram_mb: float | None = Field(default=None, ge=0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0)
    unavailable_reasons: dict[str, str] = Field(default_factory=dict)


class ExperimentManifest(ExperimentModel):
    experiment_id: str
    run_kind: Literal["pretrained", "training", "finetuned", "comparison"]
    stage: str | None = None
    model: str | None = None
    pipeline: dict[str, str | None]
    baseline_mode: str | None = None
    finetuned_mode: str | None = None
    checkpoint_source: str | None = None
    checkpoint: dict[str, Any]
    split_manifest_hash: str
    test_document_ids: list[str]
    schema_version: str
    preprocessing_config_hash: str
    postprocessing_config_hash: str
    workflow_defaults_hash: str
    validation_tolerance: str
    batch_size: int
    num_workers: int
    device: str
    hardware_fingerprint: str
    metric_code_version: str
    reproducibility: dict[str, Any]
    command_line: list[str]
    start_time: datetime
    end_time: datetime | None = None
    status: Literal["pending", "success", "failed", "skipped"] = "pending"
    skip_reason: str | None = None
