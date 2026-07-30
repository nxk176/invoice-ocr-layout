"""Canonical hashing and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def directory_manifest_hash(
    root: Path,
    suffix: str | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> str:
    if not root.exists():
        return canonical_json_hash([])
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root).as_posix()
        if any(
            relative_path == prefix or relative_path.startswith(f"{prefix}/")
            for prefix in exclude_prefixes
        ):
            continue
        if suffix is not None and path.suffix.casefold() != suffix.casefold():
            continue
        entries.append(
            {
                "relative_path": relative_path,
                "sha256": sha256_file(path),
            }
        )
    return canonical_json_hash(entries)


def file_hash_or_missing(path: Path) -> str:
    return sha256_file(path) if path.is_file() else canonical_json_hash({"missing": str(path)})


def _git_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_metadata() -> dict[str, Any]:
    status = _git_output(["status", "--porcelain"])
    return {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "branch": _git_output(["branch", "--show-current"]),
        "dirty": None if status is None else bool(status),
    }


def package_versions() -> dict[str, str | None]:
    packages = (
        "invoice-ocr-layout",
        "pydantic",
        "Pillow",
        "PyYAML",
        "jsonschema",
        "psutil",
        "torch",
        "transformers",
        "paddlepaddle",
        "paddleocr",
        "vietocr",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "operating_system": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "cuda_version": None,
        "gpu_name": None,
    }
    try:
        import torch

        metadata["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return metadata


def reproducibility_metadata() -> dict[str, Any]:
    hardware = hardware_metadata()
    return {
        "git": git_metadata(),
        "platform": hardware,
        "packages": package_versions(),
        "hardware_fingerprint": canonical_json_hash(hardware),
    }
