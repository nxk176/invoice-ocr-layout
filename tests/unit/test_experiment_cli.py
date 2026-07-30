from __future__ import annotations

from pathlib import Path

import pytest

from invoice_ocr.cli import build_parser, main, selection_from_args
from invoice_ocr.pipeline import enumerate_pipeline_combinations


@pytest.mark.parametrize(
    "command",
    ["create-split", "evaluate-model", "compare-runs", "experiment"],
)
def test_experiment_subcommands_have_help(command: str) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command, "--help"])
    assert error.value.code == 0


def test_experiment_pipeline_parser_keeps_detector_recognizer_layout_order() -> None:
    args = build_parser().parse_args(
        [
            "experiment",
            "--pipeline",
            "dbnetpp",
            "vietocr",
            "layoutlmv3",
            "--data",
            "data",
            "--gt",
            "GT",
            "--split-manifest",
            "GT/splits/split_v1.json",
            "--output",
            "outputs/experiments/synthetic",
        ]
    )
    selection = selection_from_args(args)
    assert (selection.detector, selection.recognizer, selection.layout) == (
        "dbnetpp",
        "vietocr",
        "layoutlmv3",
    )


def test_all_experiment_combinations_remain_exactly_twelve() -> None:
    args = build_parser().parse_args(
        [
            "experiment",
            "--all-combinations",
            "--data",
            "data",
            "--gt",
            "GT",
            "--output",
            "outputs/experiments/all",
        ]
    )
    assert args.all_combinations is True
    assert len(enumerate_pipeline_combinations()) == 12


def test_create_split_on_empty_data_returns_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "data"
    gt = tmp_path / "GT"
    data.mkdir()
    gt.mkdir()
    code = main(
        [
            "create-split",
            "--data",
            str(data),
            "--gt",
            str(gt),
            "--output",
            str(gt / "splits" / "split_v1.json"),
        ]
    )
    assert code == 2
    assert "no input documents found" in capsys.readouterr().err
