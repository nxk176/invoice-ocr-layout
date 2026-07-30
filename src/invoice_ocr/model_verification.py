"""Conservative readiness audit for production model backends."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from invoice_ocr.adapters.detectors import DETECTORS
from invoice_ocr.adapters.layout import LAYOUT_ADAPTERS
from invoice_ocr.adapters.recognizers import RECOGNIZERS
from invoice_ocr.model_catalog import (
    checkpoint_hash,
    checkpoint_path,
    load_model_manifest,
    verify_checkpoint,
)
from invoice_ocr.model_sources import load_source_specs, verify_source_checkout

Requirement = Literal["any", "inference", "training", "both"]


@dataclass(frozen=True)
class BackendSpec:
    key: str
    stage: str
    adapter_name: str
    manifest_name: str
    inference_dependencies: tuple[str, ...]
    training_dependencies: tuple[str, ...]
    source_name: str | None
    source_for_inference: bool
    source_for_training: bool
    inference_checkpoint: str | None
    training_checkpoint: str | None
    training_implementation_available: bool


BACKEND_SPECS: dict[str, BackendSpec] = {
    "paddleocr-detector": BackendSpec(
        "paddleocr-detector",
        "detector",
        "paddleocr",
        "paddleocr-detector",
        ("paddle", "paddleocr"),
        ("paddle",),
        "paddleocr",
        False,
        False,
        None,
        None,
        False,
    ),
    "dbnet": BackendSpec(
        "dbnet",
        "detector",
        "dbnet",
        "dbnet",
        ("torch", "cv2", "yaml", "shapely", "pyclipper"),
        ("torch", "cv2", "yaml", "shapely", "pyclipper"),
        "dbnet",
        True,
        True,
        None,
        None,
        False,
    ),
    "dbnetpp": BackendSpec(
        "dbnetpp",
        "detector",
        "dbnetpp",
        "dbnetpp",
        ("torch", "cv2", "yaml", "shapely", "pyclipper"),
        ("torch", "cv2", "yaml", "shapely", "pyclipper"),
        "dbnet",
        True,
        True,
        None,
        None,
        False,
    ),
    "paddleocr-recognizer": BackendSpec(
        "paddleocr-recognizer",
        "recognizer",
        "paddleocr",
        "paddleocr-recognizer",
        ("paddle", "paddleocr"),
        ("paddle",),
        "paddleocr",
        False,
        False,
        None,
        None,
        False,
    ),
    "vietocr": BackendSpec(
        "vietocr",
        "recognizer",
        "vietocr",
        "vietocr",
        ("torch", "vietocr"),
        ("torch", "torchvision", "vietocr"),
        None,
        False,
        False,
        None,
        None,
        False,
    ),
    "layoutlmv3": BackendSpec(
        "layoutlmv3",
        "layout",
        "layoutlmv3",
        "layoutlmv3-base",
        ("torch", "transformers"),
        ("torch", "transformers", "accelerate"),
        None,
        False,
        False,
        "layoutlmv3/invoice-best",
        None,
        True,
    ),
    "vi_layoutxlm": BackendSpec(
        "vi_layoutxlm",
        "layout",
        "vi_layoutxlm",
        "vi-layoutxlm-base",
        ("paddle", "paddlenlp"),
        ("paddle", "paddlenlp"),
        "paddleocr",
        True,
        True,
        "vi_layoutxlm/invoice-best",
        None,
        False,
    ),
}


@dataclass
class BackendReadiness:
    backend: str
    stage: str
    dependency_installed: bool
    dependency_details: dict[str, bool]
    source_checkout_required: bool
    source_checkout_found: bool | None
    expected_commit: str | None
    actual_commit: str | None
    checkpoint_required: bool
    checkpoint_found: bool
    checkpoint_path: str
    checkpoint_hash: str | None
    checkpoint_hash_valid: bool | None
    training_checkpoint_path: str
    training_checkpoint_found: bool
    inference_implementation_available: bool
    training_implementation_available: bool
    ready_for_inference: bool
    ready_for_training: bool
    reasons: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _inference_implementation_available(spec: BackendSpec) -> bool:
    if spec.stage == "detector":
        return DETECTORS[spec.adapter_name].inference_implementation_available
    if spec.stage == "recognizer":
        return RECOGNIZERS[spec.adapter_name].inference_implementation_available
    if spec.stage == "layout":
        return LAYOUT_ADAPTERS[spec.adapter_name].inference_implementation_available
    raise ValueError(f"unsupported backend stage: {spec.stage}")


def _dependencies(names: tuple[str, ...]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _checkpoint_found(path: Path, checkpoint_type: str) -> bool:
    return path.is_file() if checkpoint_type == "file" else path.is_dir()


def _verify_invoice_checkpoint(path: Path, backend: str) -> tuple[bool, str | None, str | None]:
    if not path.is_dir():
        return False, None, f"fine-tuned invoice checkpoint not found: {path}"
    if backend == "layoutlmv3":
        if not (path / "config.json").is_file():
            return (
                False,
                checkpoint_hash(path),
                f"checkpoint is incomplete; missing {path / 'config.json'}",
            )
        weights = (path / "model.safetensors", path / "pytorch_model.bin")
        if not any(candidate.is_file() for candidate in weights):
            expected = " or ".join(str(candidate) for candidate in weights)
            return False, checkpoint_hash(path), f"checkpoint is incomplete; missing {expected}"
    elif backend == "vi_layoutxlm" and not any(path.rglob("*.pdparams")):
        return (
            False,
            checkpoint_hash(path),
            f"checkpoint is incomplete; no .pdparams weights found below {path}",
        )
    return True, checkpoint_hash(path), None


def verify_backend(
    backend: str,
    model_root: Path,
    external_root: Path,
) -> BackendReadiness:
    if backend not in BACKEND_SPECS:
        raise ValueError(f"unknown backend: {backend}")
    spec = BACKEND_SPECS[backend]
    manifest = load_model_manifest(spec.manifest_name)
    inference_dependencies = _dependencies(spec.inference_dependencies)
    training_dependencies = _dependencies(spec.training_dependencies)
    dependency_details = {**training_dependencies, **inference_dependencies}

    source_found: bool | None = None
    expected_commit: str | None = None
    actual_commit: str | None = None
    source_inference_ready = not spec.source_for_inference
    source_training_ready = not spec.source_for_training
    source_reason: str | None = None
    if spec.source_name is not None:
        source_spec = load_source_specs()[spec.source_name]
        source_status = verify_source_checkout(source_spec, external_root)
        source_found = source_status.found
        expected_commit = source_status.expected_commit
        actual_commit = source_status.actual_commit
        source_inference_ready = not spec.source_for_inference or source_status.ready
        source_training_ready = not spec.source_for_training or source_status.ready
        if (spec.source_for_inference or spec.source_for_training) and not source_status.ready:
            source_reason = source_status.reason

    manifest_checkpoint = checkpoint_path(manifest, model_root)
    inference_checkpoint = (
        model_root / spec.inference_checkpoint
        if spec.inference_checkpoint is not None
        else manifest_checkpoint
    )
    if spec.inference_checkpoint is None:
        checkpoint_ok, inference_hash, checkpoint_reason = verify_checkpoint(manifest, model_root)
    else:
        checkpoint_ok, inference_hash, checkpoint_reason = _verify_invoice_checkpoint(
            inference_checkpoint, spec.key
        )
    inference_checkpoint_found = _checkpoint_found(
        inference_checkpoint,
        "directory" if spec.inference_checkpoint is not None else str(manifest["checkpoint_type"]),
    )
    training_checkpoint = (
        model_root / spec.training_checkpoint
        if spec.training_checkpoint is not None
        else manifest_checkpoint
    )
    if spec.training_checkpoint is None:
        training_checkpoint_ok, _, training_checkpoint_reason = verify_checkpoint(
            manifest, model_root
        )
        training_checkpoint_found = _checkpoint_found(
            training_checkpoint, str(manifest["checkpoint_type"])
        )
    else:
        training_checkpoint_ok = training_checkpoint.exists()
        training_checkpoint_found = training_checkpoint_ok
        training_checkpoint_reason = (
            None
            if training_checkpoint_ok
            else f"training checkpoint not found: {training_checkpoint}"
        )

    inference_implementation = _inference_implementation_available(spec)
    ready_for_inference = (
        inference_implementation
        and all(inference_dependencies.values())
        and source_inference_ready
        and checkpoint_ok
    )
    ready_for_training = (
        spec.training_implementation_available
        and all(training_dependencies.values())
        and source_training_ready
        and training_checkpoint_ok
    )
    reasons: list[str] = []
    missing_inference = [name for name, found in inference_dependencies.items() if not found]
    missing_training = [name for name, found in training_dependencies.items() if not found]
    if missing_inference:
        reasons.append(f"missing inference dependencies: {', '.join(missing_inference)}")
    if missing_training:
        reasons.append(f"missing training dependencies: {', '.join(missing_training)}")
    if source_reason:
        reasons.append(source_reason)
    if checkpoint_reason:
        reasons.append(checkpoint_reason)
    if training_checkpoint_reason and training_checkpoint_reason != checkpoint_reason:
        reasons.append(training_checkpoint_reason)
    if not inference_implementation:
        reasons.append("production inference adapter is an audited scaffold")
    if not spec.training_implementation_available:
        reasons.append(
            "production training path lacks complete dataset preparation, resume, and "
            "validation-only checkpoint selection"
        )

    return BackendReadiness(
        backend=backend,
        stage=spec.stage,
        dependency_installed=all(dependency_details.values()),
        dependency_details=dependency_details,
        source_checkout_required=spec.source_for_inference or spec.source_for_training,
        source_checkout_found=source_found,
        expected_commit=expected_commit,
        actual_commit=actual_commit,
        checkpoint_required=True,
        checkpoint_found=inference_checkpoint_found,
        checkpoint_path=str(inference_checkpoint),
        checkpoint_hash=inference_hash,
        checkpoint_hash_valid=(checkpoint_ok if manifest.get("sha256") is not None else None),
        training_checkpoint_path=str(training_checkpoint),
        training_checkpoint_found=training_checkpoint_found,
        inference_implementation_available=inference_implementation,
        training_implementation_available=spec.training_implementation_available,
        ready_for_inference=ready_for_inference,
        ready_for_training=ready_for_training,
        reasons=list(dict.fromkeys(reasons)),
    )


def readiness_satisfies(status: BackendReadiness, requirement: Requirement) -> bool:
    if requirement == "inference":
        return status.ready_for_inference
    if requirement == "training":
        return status.ready_for_training
    if requirement == "both":
        return status.ready_for_inference and status.ready_for_training
    return status.ready_for_inference or status.ready_for_training


def verify_backends(
    backends: list[str],
    model_root: Path,
    external_root: Path,
) -> list[BackendReadiness]:
    return [verify_backend(backend, model_root, external_root) for backend in backends]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify production model backend readiness.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument(
        "--backend",
        action="append",
        choices=tuple(BACKEND_SPECS),
        help="repeat to verify multiple backends",
    )
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--external-root", type=Path, default=Path("external"))
    parser.add_argument(
        "--require",
        choices=("any", "inference", "training", "both"),
        default="any",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def run_verification(args: argparse.Namespace) -> int:
    selected = list(BACKEND_SPECS) if args.all else list(args.backend)
    statuses = verify_backends(selected, args.model_root, args.external_root)
    if args.json_output:
        print(json.dumps([status.as_dict() for status in statuses], indent=2))
    else:
        for status in statuses:
            ready = readiness_satisfies(status, args.require)
            state = "READY" if ready else "NOT_READY"
            reason = "; ".join(status.reasons) if status.reasons else "all checks passed"
            print(
                f"{state}: {status.backend}: inference={status.ready_for_inference}, "
                f"training={status.ready_for_training}: {reason}"
            )
    return 0 if all(readiness_satisfies(status, args.require) for status in statuses) else 1
