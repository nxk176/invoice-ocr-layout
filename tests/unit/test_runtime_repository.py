from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from invoice_ocr.cli import build_parser, main
from invoice_ocr.exceptions import InvalidGroundTruthError, NoInputDocumentsError
from invoice_ocr.io.paths import (
    RUNTIME_DIRECTORY_NAMES,
    discover_documents,
    ensure_runtime_directories,
)
from invoice_ocr.training.datasets import ensure_stage_annotations, validate_ground_truth

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOTS = frozenset(RUNTIME_DIRECTORY_NAMES)


def tracked_paths() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    return [Path(value.decode()) for value in completed.stdout.split(b"\0") if value]


def test_runtime_directories_can_start_absent_and_are_created_idempotently(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "fresh-clone"
    assert not project_root.exists()

    first = ensure_runtime_directories(project_root)
    second = ensure_runtime_directories(project_root)

    assert first == second
    assert {path.name for path in first} == RUNTIME_ROOTS
    assert all(path.is_dir() for path in first)


def test_cli_creates_default_runtime_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["validate-gt", "--gt", str(tmp_path / "private-gt")])

    assert exit_code == 2
    assert all((tmp_path / name).is_dir() for name in RUNTIME_DIRECTORY_NAMES)


def test_missing_data_has_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(NoInputDocumentsError, match="no input documents found"):
        discover_documents(tmp_path / "missing-data")


def test_missing_ground_truth_has_friendly_error(tmp_path: Path) -> None:
    missing_gt = tmp_path / "missing-gt"
    report = validate_ground_truth(missing_gt)

    assert not report.is_valid
    assert report.errors
    assert "ground-truth directory does not exist" in report.errors[0]
    with pytest.raises(InvalidGroundTruthError, match="ground-truth directory does not exist"):
        ensure_stage_annotations(missing_gt, "detector")


def test_setup_scripts_declare_idempotent_runtime_directory_creation() -> None:
    local_setup = (PROJECT_ROOT / "scripts" / "setup_local.ps1").read_text(encoding="utf-8")
    assert "New-Item -ItemType Directory -Force" in local_setup

    bash_scripts = (
        "setup_server.sh",
        "run_server.sh",
        "train_server.sh",
        "benchmark_server.sh",
        "run_experiment_server.sh",
    )
    for name in bash_scripts:
        content = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "mkdir -p" in content
        for directory in RUNTIME_DIRECTORY_NAMES:
            assert f'"$project_root/{directory}"' in content


def test_runtime_roots_are_fully_ignored_without_placeholder_exceptions() -> None:
    lines = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert {f"/{directory}/" for directory in RUNTIME_DIRECTORY_NAMES} <= lines
    assert not any(
        line.startswith(f"!{directory}/") for directory in RUNTIME_DIRECTORY_NAMES for line in lines
    )

    for directory in RUNTIME_DIRECTORY_NAMES:
        completed = subprocess.run(
            ["git", "check-ignore", "--no-index", f"{directory}/private-artifact.json"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_git_tracks_no_runtime_placeholder_or_artifact() -> None:
    violations = [path for path in tracked_paths() if path.parts and path.parts[0] in RUNTIME_ROOTS]
    assert violations == []


def test_synthetic_fixture_is_confined_to_test_fixture_directory() -> None:
    synthetic = [path for path in tracked_paths() if path.name.startswith("synthetic_")]
    assert synthetic
    assert all(path.parts[:2] == ("tests", "fixtures") for path in synthetic)


def test_core_pipeline_and_experiment_parser_contracts_are_unchanged() -> None:
    parser = build_parser()
    run_args = parser.parse_args(
        [
            "run",
            "--pipeline",
            "paddleocr",
            "vietocr",
            "layoutlmv3",
            "--input",
            "data",
            "--output",
            "outputs/run",
        ]
    )
    experiment_args = parser.parse_args(
        [
            "experiment",
            "--all-combinations",
            "--protocol",
            "pretrained-vs-finetuned",
            "--data",
            "data",
            "--gt",
            "GT",
            "--output",
            "outputs/experiments/all",
        ]
    )

    assert run_args.pipeline == ["paddleocr", "vietocr", "layoutlmv3"]
    assert experiment_args.all_combinations is True
    assert experiment_args.protocol == "pretrained-vs-finetuned"
