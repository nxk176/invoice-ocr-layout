"""Download and verify revision-pinned public base checkpoints."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invoice_ocr.model_catalog import (  # noqa: E402
    available_model_names,
    checkpoint_path,
    load_model_manifest,
    sha256_file,
    verify_checkpoint,
)

LOGGER = logging.getLogger("download_models")
MANIFEST_ROOT = PROJECT_ROOT / "configs" / "models"


def load_manifest(name: str) -> dict[str, Any]:
    """Compatibility wrapper around the shared validated manifest loader."""
    return load_model_manifest(name, MANIFEST_ROOT)


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


def _copy_member(source: BinaryIO, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as output:
        shutil.copyfileobj(source, output)


def safe_extract_tar(archive: Path, destination: Path, strip_components: int = 0) -> None:
    """Extract ordinary files only, optionally removing a common archive prefix."""
    if strip_components < 0:
        raise ValueError("strip_components must be non-negative")
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name}")
            parts = PurePosixPath(member.name).parts
            if len(parts) <= strip_components:
                continue
            relative = Path(*parts[strip_components:])
            target = (destination / relative).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"unsafe archive member path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member type: {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"could not read archive member: {member.name}")
            with source:
                _copy_member(source, target)


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


def _write_download_digest(destination: Path, target: Path, digest: str) -> None:
    digest_path = (
        destination / "DOWNLOAD.sha256"
        if destination.is_dir()
        else destination.with_suffix(destination.suffix + ".sha256")
    )
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(f"{digest}  {target.name}\n", encoding="utf-8")


def download_model(manifest: dict[str, Any], model_root: Path, force: bool) -> str:
    destination = checkpoint_path(manifest, model_root)
    if destination.exists() and not force:
        valid, digest, reason = verify_checkpoint(manifest, model_root)
        if not valid:
            raise ValueError(
                f"existing checkpoint for {manifest['model_name']} is invalid: {reason}; "
                "inspect it or rerun with --force"
            )
        return f"kept verified {destination} (sha256={digest})"

    url = manifest.get("url")
    if not url:
        setup = manifest.get("manual_setup", "follow the official repository instructions")
        return f"manual setup required for {manifest['model_name']}: {setup}"

    archive_type = manifest.get("archive")
    if archive_type == "huggingface_snapshot":
        download_snapshot(manifest, destination)
        valid, digest, reason = verify_checkpoint(manifest, model_root)
        if not valid:
            raise ValueError(f"downloaded snapshot is incomplete: {reason}")
        return f"downloaded verified snapshot to {destination} (sha256={digest})"

    target = (
        model_root / ".downloads" / Path(str(url)).name if archive_type == "tar" else destination
    )
    digest = download_file(str(url), target)
    expected = manifest.get("sha256")
    if expected and digest != str(expected):
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
        safe_extract_tar(
            target,
            destination,
            strip_components=int(manifest.get("strip_components", 0)),
        )
    _write_download_digest(destination, target, digest)
    valid, checkpoint_digest, reason = verify_checkpoint(manifest, model_root)
    if not valid:
        raise ValueError(f"downloaded checkpoint is incomplete: {reason}")
    return (
        f"downloaded verified {manifest['model_name']} to {destination} "
        f"(sha256={checkpoint_digest})"
    )


def verify_models(names: list[str], model_root: Path) -> int:
    failed = False
    for name in names:
        manifest = load_manifest(name)
        ready, digest, reason = verify_checkpoint(manifest, model_root)
        if ready:
            print(f"READY {manifest['model_name']}: sha256={digest}")
        else:
            failed = True
            print(f"NOT_READY {manifest['model_name']}: {reason}")
    return 2 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or verify revision-pinned public base models; never invoice weights."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--model",
        choices=available_model_names(include_aliases=True),
    )
    selection.add_argument("--all", action="store_true")
    selection.add_argument(
        "--verify",
        action="store_true",
        help="verify every declared checkpoint without downloading",
    )
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--force", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    model_root: Path = args.model_root
    model_root.mkdir(parents=True, exist_ok=True)
    names = list(available_model_names()) if args.all or args.verify else [args.model]
    try:
        if args.verify:
            return verify_models(names, model_root)
        for name in names:
            print(download_model(load_manifest(name), model_root, args.force))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
