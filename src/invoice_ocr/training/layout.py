"""Layout/KIE dataset encoding and fine-tuning entry points."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TypedDict

import yaml

from invoice_ocr.exceptions import CheckpointUnavailableError, DependencyUnavailableError
from invoice_ocr.training.datasets import align_word_labels, load_layout_pages


class LayoutTrainerProgressArguments(TypedDict):
    """Progress settings passed directly to Transformers TrainingArguments."""

    disable_tqdm: bool
    logging_strategy: str
    logging_steps: int
    logging_first_step: bool


def layout_trainer_progress_arguments(
    training_samples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    target_logs_per_epoch: int = 20,
) -> LayoutTrainerProgressArguments:
    """Return deterministic step logging settings for a live progress bar."""
    values = (
        ("training_samples", training_samples),
        ("per_device_train_batch_size", per_device_train_batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("target_logs_per_epoch", target_logs_per_epoch),
    )
    invalid = [name for name, value in values if value <= 0]
    if invalid:
        raise ValueError(f"layout training progress values must be positive: {invalid}")
    batches_per_epoch = math.ceil(training_samples / per_device_train_batch_size)
    optimizer_steps = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    return {
        "disable_tqdm": False,
        "logging_strategy": "steps",
        "logging_steps": max(1, math.ceil(optimizer_steps / target_logs_per_epoch)),
        "logging_first_step": True,
    }


def load_bio_label_mapping(path: Path) -> tuple[list[str], dict[str, int]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_labels = [str(label) for label in loaded["labels"]]
    labels = ["O"]
    for label in base_labels:
        if label != "O":
            labels.extend([f"B-{label}", f"I-{label}"])
    return labels, {label: index for index, label in enumerate(labels)}


class LayoutPageDataset:
    """Lazily encode annotated pages for Hugging Face Trainer."""

    def __init__(
        self,
        pages: list[dict[str, Any]],
        processor: Any,
        label_to_id: dict[str, int],
        image_root: Path,
    ) -> None:
        self.pages = pages
        self.processor = processor
        self.label_to_id = label_to_id
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        page = self.pages[index]
        image_value = page.get("image_path")
        if not image_value:
            raise ValueError(
                f"layout page {page['document_id']}:{page['page_index']} lacks image_path"
            )
        image_path = self.image_root / str(image_value)
        if not image_path.is_file():
            raise FileNotFoundError(f"layout training image not found: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            encoding = self.processor(
                images=image,
                text=page["tokens"],
                boxes=page["boxes"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        ignore_mask = page.get("ignore_mask", [False] * len(page["labels"]))
        label_ids = [
            -100 if ignore else self.label_to_id[str(label)]
            for label, ignore in zip(page["labels"], ignore_mask, strict=True)
        ]
        encoding["labels"] = align_word_labels(encoding, label_ids)
        return {
            key: value.squeeze(0) if hasattr(value, "squeeze") else value
            for key, value in encoding.items()
        }


def train_layoutlmv3(
    annotation_paths: list[Path],
    data_root: Path,
    model_root: Path,
    output_dir: Path,
    seed: int,
    resume: bool,
    device: str,
) -> None:
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoProcessor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise DependencyUnavailableError(
            "LayoutLMv3 training requires torch and the pinned transformers package."
        ) from exc
    base_checkpoint = model_root / "layoutlmv3-base"
    if not base_checkpoint.is_dir():
        raise CheckpointUnavailableError(
            f"LayoutLMv3 base checkpoint not found at {base_checkpoint}. Run the model "
            "downloader before fine-tuning."
        )
    labels, label_to_id = load_bio_label_mapping(Path("configs/labels/invoice_bio_labels.yaml"))
    processor = AutoProcessor.from_pretrained(base_checkpoint, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(
        base_checkpoint,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    pages = load_layout_pages(annotation_paths)
    dataset = LayoutPageDataset(pages, processor, label_to_id, data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 1
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        **layout_trainer_progress_arguments(
            len(dataset),
            per_device_train_batch_size,
            gradient_accumulation_steps,
        ),
        save_strategy="steps",
        evaluation_strategy="no",
        load_best_model_at_end=False,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir / "invoice-best")
    processor.save_pretrained(output_dir / "invoice-best")
    (output_dir / "training_metrics.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
    )


def train_vi_layoutxlm(
    annotation_paths: list[Path],
    data_root: Path,
    model_root: Path,
    output_dir: Path,
    seed: int,
    resume: bool,
    device: str,
) -> None:
    raise DependencyUnavailableError(
        "VI-LayoutXLM training uses the revision-pinned official PaddleOCR "
        "PP-Structure KIE command. Convert validated layout annotations with the documented "
        "dataset preparation command, then launch the official config; no generic substitute "
        "is invoked automatically."
    )
