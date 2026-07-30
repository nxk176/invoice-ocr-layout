from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import cast

import pytest

from invoice_ocr.adapters.detectors.base import DetectorAdapter
from invoice_ocr.adapters.detectors.dbnet import DBNetDetector
from invoice_ocr.adapters.detectors.dbnetpp import DBNetPPDetector
from invoice_ocr.adapters.layout.vi_layoutxlm import VILayoutXLMAdapter
from invoice_ocr.cli import build_parser, main
from invoice_ocr.contracts import DocumentPage
from invoice_ocr.exceptions import DependencyUnavailableError
from invoice_ocr.model_catalog import load_model_manifest, sha256_file, verify_checkpoint
from invoice_ocr.model_verification import verify_backend
from invoice_ocr.setup_selection import select_all_models_setup, select_pipeline_setup


def _paddle_checkpoint(model_root: Path) -> Path:
    checkpoint = model_root / "paddleocr" / "detector"
    checkpoint.mkdir(parents=True)
    (checkpoint / "inference.pdmodel").write_bytes(b"synthetic model")
    (checkpoint / "inference.pdiparams").write_bytes(b"synthetic parameters")
    return checkpoint


def test_missing_dependency_keeps_backend_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "models"
    _paddle_checkpoint(model_root)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    status = verify_backend("paddleocr-detector", model_root, tmp_path / "external")

    assert status.inference_implementation_available
    assert not status.dependency_installed
    assert not status.ready_for_inference
    assert any("missing inference dependencies" in reason for reason in status.reasons)


def test_missing_source_and_checkpoint_are_reported(tmp_path: Path) -> None:
    status = verify_backend("dbnetpp", tmp_path / "models", tmp_path / "external")

    assert status.source_checkout_required
    assert status.source_checkout_found is False
    assert not status.checkpoint_found
    assert not status.ready_for_inference
    assert any("source checkout not found" in reason for reason in status.reasons)
    assert any("checkpoint file not found" in reason for reason in status.reasons)


def test_wrong_checkpoint_hash_returns_digest_and_reason(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"not the expected bytes")
    manifest = {
        "checkpoint_type": "file",
        "local_path": "checkpoint.bin",
        "sha256": "0" * 64,
    }

    ready, digest, reason = verify_checkpoint(manifest, tmp_path)

    assert not ready
    assert digest == sha256_file(checkpoint)
    assert reason is not None and "SHA-256 mismatch" in reason


def test_verify_models_cli_returns_nonzero_for_unready_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "verify-models",
            "--backend",
            "dbnetpp",
            "--model-root",
            str(tmp_path / "models"),
            "--external-root",
            str(tmp_path / "external"),
        ]
    )

    assert exit_code == 1


def test_verify_models_parser_accepts_all_and_individual_backend() -> None:
    parser = build_parser()

    all_args = parser.parse_args(["verify-models", "--all"])
    one_args = parser.parse_args(["verify-models", "--backend", "layoutlmv3"])

    assert all_args.all is True
    assert one_args.backend == ["layoutlmv3"]


def test_production_scaffolds_are_never_marked_ready() -> None:
    assert not DBNetDetector.inference_implementation_available
    assert not DBNetPPDetector.inference_implementation_available
    assert not VILayoutXLMAdapter.inference_implementation_available


def test_dbnet_adapter_never_falls_back_to_mock(tmp_path: Path) -> None:
    adapter: DetectorAdapter = DBNetDetector(tmp_path)

    with pytest.raises(DependencyUnavailableError, match="official MhLiao/DB"):
        adapter.detect(cast(DocumentPage, None))


def test_pipeline_specific_setup_selects_only_three_backends() -> None:
    plan = select_pipeline_setup("paddleocr", "vietocr", "layoutlmv3")

    assert plan.backends == ("paddleocr-detector", "vietocr", "layoutlmv3")
    assert plan.models == ("paddleocr-detector", "vietocr", "layoutlmv3-base")
    assert plan.sources == ()
    assert plan.extras == ("paddleocr", "vietocr", "layoutlmv3")


def test_all_models_setup_includes_all_audited_backends() -> None:
    plan = select_all_models_setup()

    assert len(plan.backends) == 7
    assert plan.sources == ("paddleocr", "dbnet")


@pytest.mark.parametrize(
    "name",
    [
        "paddleocr-detector",
        "paddleocr-recognizer",
        "dbnet",
        "dbnetpp",
        "vietocr",
        "layoutlmv3-base",
        "vi-layoutxlm",
    ],
)
def test_model_manifests_have_verifiable_checkpoint_contract(name: str) -> None:
    manifest = load_model_manifest(name)

    assert manifest["pretrained"] is True
    assert manifest["fine_tuned_for_invoice"] is False
    assert len(manifest["revision"]) == 40


def test_scaffold_stays_unready_with_dependency_source_and_checkpoint_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import invoice_ocr.model_verification as verification

    checkpoint = tmp_path / "models" / "dbnet" / "totaltext_resnet50"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic official-style checkpoint")
    monkeypatch.setattr(
        verification,
        "_dependencies",
        lambda names: {name: True for name in names},
    )
    monkeypatch.setattr(
        verification,
        "verify_source_checkout",
        lambda _spec, _root: SimpleNamespace(
            found=True,
            ready=True,
            expected_commit="65ca77a0bcfbd7114b916cf8a1e9ca85114286ce",
            actual_commit="65ca77a0bcfbd7114b916cf8a1e9ca85114286ce",
            reason="source checkout matches",
        ),
    )

    status = verification.verify_backend("dbnet", tmp_path / "models", tmp_path / "external")

    assert status.checkpoint_found
    assert status.source_checkout_found
    assert not status.inference_implementation_available
    assert not status.ready_for_inference
    assert any("audited scaffold" in reason for reason in status.reasons)


def test_setup_script_orders_install_fetch_download_and_verify() -> None:
    project_root = Path(__file__).resolve().parents[2]
    setup = (project_root / "scripts" / "setup_server.sh").read_text(encoding="utf-8")

    install_index = setup.index('python -m pip install -e "$specifier"')
    fetch_index = setup.index("fetch_model_sources.py")
    download_index = setup.index("download_models.py")
    verify_index = setup.index("verify-models")

    assert install_index < fetch_index < download_index < verify_index


def test_auto_device_checks_paddle_when_torch_has_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    from invoice_ocr.pipeline import resolve_device

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(
        sys.modules,
        "paddle",
        SimpleNamespace(is_compiled_with_cuda=lambda: True),
    )

    assert resolve_device("auto") == "cuda"
    assert resolve_device("cpu") == "cpu"
