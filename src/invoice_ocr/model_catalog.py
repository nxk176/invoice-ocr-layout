"""Validated model-manifest catalog and local checkpoint inspection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_MANIFEST_ROOT = PROJECT_ROOT / "configs" / "models"
MODEL_ALIASES = {"vi-layoutxlm": "vi-layoutxlm-base"}
MODEL_MANIFEST_NAMES = (
    "paddleocr-detector",
    "paddleocr-recognizer",
    "dbnet",
    "dbnetpp",
    "vietocr",
    "layoutlmv3-base",
    "vi-layoutxlm-base",
)

REQUIRED_MANIFEST_FIELDS = {
    "model_name",
    "backend",
    "task",
    "framework",
    "official_repository",
    "revision",
    "checkpoint_identifier",
    "url",
    "local_path",
    "sha256",
    "license",
    "expected_task",
    "expected_config",
    "pretrained",
    "fine_tuned_for_invoice",
    "checkpoint_type",
}


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def available_model_names(include_aliases: bool = False) -> tuple[str, ...]:
    names = MODEL_MANIFEST_NAMES
    if include_aliases:
        return (*names, *MODEL_ALIASES)
    return names


def load_model_manifest(name: str, manifest_root: Path = MODEL_MANIFEST_ROOT) -> dict[str, Any]:
    canonical = canonical_model_name(name)
    path = manifest_root / f"{canonical}.yaml"
    if not path.is_file():
        available = ", ".join(available_model_names(include_aliases=True))
        raise ValueError(f"unknown model '{name}'. Available manifests: {available}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in model manifest {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"model manifest must be a YAML mapping: {path}")
    missing = sorted(REQUIRED_MANIFEST_FIELDS - loaded.keys())
    if missing:
        raise ValueError(f"manifest {path} is missing fields: {', '.join(missing)}")
    revision = str(loaded["revision"])
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError(f"manifest {path} revision must be an exact lowercase Git commit")
    if loaded["checkpoint_type"] not in {"file", "directory"}:
        raise ValueError(f"manifest {path} checkpoint_type must be file or directory")
    if loaded.get("fine_tuned_for_invoice") is True:
        raise ValueError("public base-model catalog refuses invoice fine-tuned checkpoints")
    expected_hash = loaded.get("sha256")
    if expected_hash is not None:
        digest = str(expected_hash)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"manifest {path} sha256 must be null or 64 lowercase hex digits")
    return loaded


def checkpoint_path(manifest: dict[str, Any], model_root: Path) -> Path:
    return model_root / str(manifest["local_path"])


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_hash(path: Path) -> str | None:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return sha256_directory(path)
    return None


def verify_checkpoint(
    manifest: dict[str, Any],
    model_root: Path,
    path_override: Path | None = None,
) -> tuple[bool, str | None, str | None]:
    path = path_override or checkpoint_path(manifest, model_root)
    expected_type = str(manifest["checkpoint_type"])
    if expected_type == "file" and not path.is_file():
        return False, None, f"checkpoint file not found: {path}"
    if expected_type == "directory" and not path.is_dir():
        return False, None, f"checkpoint directory not found: {path}"
    for relative in manifest.get("required_files") or []:
        if not (path / str(relative)).is_file():
            return False, None, f"checkpoint is incomplete; missing {path / str(relative)}"
    actual_hash = checkpoint_hash(path)
    expected_hash = manifest.get("sha256")
    if expected_hash is not None and actual_hash != str(expected_hash):
        return (
            False,
            actual_hash,
            f"checkpoint SHA-256 mismatch: expected {expected_hash}, found {actual_hash}",
        )
    return True, actual_hash, None
