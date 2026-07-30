"""Training request validation, split persistence, and backend dispatch."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from invoice_ocr.exceptions import ConfigurationError
from invoice_ocr.training.datasets import ensure_stage_annotations
from invoice_ocr.training.detector import train_detector
from invoice_ocr.training.layout import train_layoutlmv3, train_vi_layoutxlm
from invoice_ocr.training.recognizer import train_recognizer
from invoice_ocr.training.splits import create_split_manifest


@dataclass
class TrainingRequest:
    stage: str
    model: str
    data_root: Path
    gt_root: Path
    model_root: Path = Path("models")
    output_root: Path = Path("outputs/training")
    device: str = "auto"
    seed: int = 42
    resume: bool = False
    force: bool = False


def run_training(request: TrainingRequest) -> Path:
    if request.stage not in {"detector", "recognizer", "layout"}:
        raise ConfigurationError(f"unsupported training stage: {request.stage}")
    annotations = ensure_stage_annotations(request.gt_root, request.stage)
    random.seed(request.seed)
    output_dir = request.output_root / f"{request.stage}-{request.model}"
    if (output_dir / "training_complete.json").is_file() and not (request.resume or request.force):
        raise ConfigurationError(
            f"completed training output exists at {output_dir}; use --resume or --force"
        )
    document_ids = [path.stem for path in annotations]
    create_split_manifest(
        document_ids, output_dir / "split_manifest.json", request.seed, request.resume
    )
    if request.stage == "detector":
        train_detector(
            request.model,
            annotations,
            request.data_root,
            output_dir,
            request.resume,
            request.device,
        )
    elif request.stage == "recognizer":
        train_recognizer(
            request.model,
            annotations,
            request.data_root,
            output_dir,
            request.resume,
            request.device,
        )
    elif request.model == "layoutlmv3":
        train_layoutlmv3(
            annotations,
            request.data_root,
            request.model_root,
            output_dir,
            request.seed,
            request.resume,
            request.device,
        )
    elif request.model == "vi_layoutxlm":
        train_vi_layoutxlm(
            annotations,
            request.data_root,
            request.model_root,
            output_dir,
            request.seed,
            request.resume,
            request.device,
        )
    else:
        raise ConfigurationError(f"unsupported layout training model: {request.model}")
    completion = {
        "stage": request.stage,
        "model": request.model,
        "seed": request.seed,
        "annotation_count": len(annotations),
    }
    (output_dir / "training_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def run_pipeline_training(
    detector: str,
    recognizer: str,
    layout: str,
    data_root: Path,
    gt_root: Path,
    model_root: Path,
    output_root: Path,
    device: str,
    seed: int,
    resume: bool,
    force: bool,
) -> list[Path]:
    stages = [
        ("detector", detector),
        ("recognizer", recognizer),
        ("layout", layout),
    ]
    return [
        run_training(
            TrainingRequest(
                stage=stage,
                model=model,
                data_root=data_root,
                gt_root=gt_root,
                model_root=model_root,
                output_root=output_root,
                device=device,
                seed=seed,
                resume=resume,
                force=force,
            )
        )
        for stage, model in stages
    ]
