"""Implementation for the public base-model manifest downloader."""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("download_models")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = PROJECT_ROOT / "configs" / "models"


def load_manifest(name: str) -> dict[str, Any]:
    path = MANIFEST_ROOT / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in MANIFEST_ROOT.glob("*.yaml")))
        raise ValueError(f"unknown model '{name}'. Available manifests: {available}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"model manifest must be a YAML mapping: {path}")
    required = {
        "model_name",
        "backend",
        "official_repository",
        "revision",
        "checkpoint_identifier",
        "local_path",
        "license",
        "expected_task",
    }
    missing = sorted(required - loaded.keys())
    if missing:
        raise ValueError(f"manifest {path} is missing fields: {', '.join(missing)}")
    if loaded.get("fine_tuned_for_invoice") is True:
        raise ValueError("downloader refuses invoice fine-tuned checkpoints")
    return loaded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "invoice-ocr-layout/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    except (OSError, urllib.error.URLError) as exc:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    partial.replace(destination)
    return sha256_file(destination)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Validate every member before extraction; compatible with Python 3.10."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise ValueError(f"unsafe archive member path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
        bundle.extractall(destination, members=members)


def download_snapshot(manifest: dict[str, Any], destination: Path) -> None:
    identifier = str(manifest["checkpoint_identifier"])
    revision = str(manifest["checkpoint_revision"])
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Hugging Face snapshot manifest requires a non-empty files list")
    for filename in files:
        url = f"https://huggingface.co/{identifier}/resolve/{revision}/{filename}?download=true"
        target = destination / str(filename)
        digest = download_file(url, target)
        LOGGER.info("Downloaded %s (sha256=%s)", target, digest)


def download_model(manifest: dict[str, Any], model_root: Path, force: bool) -> str:
    destination = model_root / str(manifest["local_path"])
    if destination.exists() and not force:
        return f"kept existing {destination}"
    url = manifest.get("url")
    if not url:
        setup = manifest.get("manual_setup", "follow the official repository instructions")
        return f"manual setup required for {manifest['model_name']}: {setup}"
    archive_type = manifest.get("archive")
    if archive_type == "huggingface_snapshot":
        download_snapshot(manifest, destination)
        return f"downloaded snapshot to {destination}"
    target = (
        model_root / ".downloads" / Path(str(url)).name if archive_type == "tar" else destination
    )
    digest = download_file(str(url), target)
    expected = manifest.get("sha256")
    if expected and digest.casefold() != str(expected).casefold():
        target.unlink()
        raise ValueError(
            f"checksum mismatch for {manifest['model_name']}: expected {expected}, got {digest}"
        )
    if not expected:
        LOGGER.warning(
            "No upstream SHA-256 was published for %s; downloaded digest is %s",
            manifest["model_name"],
            digest,
        )
    if archive_type == "tar":
        safe_extract_tar(target, destination)
    digest_path = (
        destination / "DOWNLOAD.sha256"
        if destination.is_dir()
        else destination.with_suffix(destination.suffix + ".sha256")
    )
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return f"downloaded {manifest['model_name']} to {destination}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download revision-pinned public base models; never invoice weights."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--force", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    names = sorted(path.stem for path in MANIFEST_ROOT.glob("*.yaml")) if args.all else [args.model]
    try:
        for name in names:
            print(download_model(load_manifest(name), args.model_root, args.force))
        return 0
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        LOGGER.error("%s", exc)
        return 2
