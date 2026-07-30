from __future__ import annotations

from pathlib import Path

import pytest

from invoice_ocr.cli import build_parser, main, selection_from_args
from invoice_ocr.evaluation.benchmark import benchmark_rows
from invoice_ocr.pipeline import enumerate_pipeline_combinations


def test_explicit_pipeline_parser_order() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--detector",
            "paddleocr",
            "--recognizer",
            "vietocr",
            "--layout",
            "layoutlmv3",
            "--input",
            "data",
            "--output",
            "outputs/run",
        ]
    )
    selection = selection_from_args(args)
    assert (selection.detector, selection.recognizer, selection.layout) == (
        "paddleocr",
        "vietocr",
        "layoutlmv3",
    )


def test_pipeline_shortcut_parser_order() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--pipeline",
            "dbnetpp",
            "paddleocr",
            "vi_layoutxlm",
            "--input",
            "data",
            "--output",
            "outputs/run",
        ]
    )
    selection = selection_from_args(args)
    assert (selection.detector, selection.recognizer, selection.layout) == (
        "dbnetpp",
        "paddleocr",
        "vi_layoutxlm",
    )


def test_all_combinations_is_exact_cartesian_product() -> None:
    combinations = enumerate_pipeline_combinations()
    assert len(combinations) == 12
    assert len(set(combinations)) == 12
    assert len(benchmark_rows(True)) == 12


@pytest.mark.parametrize(
    "command",
    [
        "run",
        "benchmark",
        "detect",
        "recognize",
        "extract",
        "postprocess",
        "evaluate",
        "train",
        "train-pipeline",
        "validate-gt",
    ],
)
def test_every_subcommand_has_help(command: str) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command, "--help"])
    assert error.value.code == 0


def test_empty_input_cli_returns_clean_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "run",
            "--pipeline",
            "paddleocr",
            "vietocr",
            "layoutlmv3",
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert code == 2
    assert "no input documents found" in capsys.readouterr().err
