from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_downloader() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "model_downloader.py"
    spec = importlib.util.spec_from_file_location("test_model_downloader_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tar_download_is_flattened_to_declared_checkpoint_directory(tmp_path: Path) -> None:
    downloader = _load_downloader()
    archive = tmp_path / "model.tar"
    content = b"synthetic paddle model"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("official_model/inference.pdmodel")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))

    destination = tmp_path / "checkpoint"
    downloader.safe_extract_tar(archive, destination, strip_components=1)

    assert (destination / "inference.pdmodel").read_bytes() == content
    assert not (destination / "official_model").exists()


def test_tar_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    downloader = _load_downloader()
    archive = tmp_path / "unsafe.tar"
    content = b"unsafe"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("../outside.bin")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))

    destination = tmp_path / "checkpoint"
    try:
        downloader.safe_extract_tar(archive, destination)
    except ValueError as exc:
        assert "unsafe archive member path" in str(exc)
    else:
        raise AssertionError("path traversal archive should have been rejected")


def test_downloader_accepts_vi_layoutxlm_alias() -> None:
    downloader = _load_downloader()

    args = downloader.build_parser().parse_args(["--model", "vi-layoutxlm"])
    manifest = downloader.load_manifest(args.model)

    assert manifest["model_name"] == "vi-layoutxlm-base"


def test_downloader_verify_returns_nonzero_for_missing_checkpoint(tmp_path: Path) -> None:
    downloader = _load_downloader()

    exit_code = downloader.verify_models(["vietocr"], tmp_path / "models")

    assert exit_code == 2
