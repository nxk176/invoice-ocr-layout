"""Deterministic dependency/source/checkpoint plan for server setup."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

DETECTORS = ("paddleocr", "dbnet", "dbnetpp")
RECOGNIZERS = ("paddleocr", "vietocr")
LAYOUT_MODELS = ("layoutlmv3", "vi_layoutxlm")


@dataclass(frozen=True)
class SetupSelection:
    extras: tuple[str, ...]
    sources: tuple[str, ...]
    models: tuple[str, ...]
    backends: tuple[str, ...]


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def select_pipeline_setup(detector: str, recognizer: str, layout: str) -> SetupSelection:
    if detector not in DETECTORS:
        raise ValueError(f"unsupported detector: {detector}")
    if recognizer not in RECOGNIZERS:
        raise ValueError(f"unsupported recognizer: {recognizer}")
    if layout not in LAYOUT_MODELS:
        raise ValueError(f"unsupported layout model: {layout}")

    extras: list[str] = []
    sources: list[str] = []
    models: list[str] = []
    backends: list[str] = []

    if detector == "paddleocr":
        extras.append("paddleocr")
        models.append("paddleocr-detector")
        backends.append("paddleocr-detector")
    else:
        extras.append("dbnet")
        sources.append("dbnet")
        models.append(detector)
        backends.append(detector)

    if recognizer == "paddleocr":
        extras.append("paddleocr")
        models.append("paddleocr-recognizer")
        backends.append("paddleocr-recognizer")
    else:
        extras.append("vietocr")
        models.append("vietocr")
        backends.append("vietocr")

    if layout == "layoutlmv3":
        extras.append("layoutlmv3")
        models.append("layoutlmv3-base")
        backends.append("layoutlmv3")
    else:
        extras.append("vi_layoutxlm")
        sources.append("paddleocr")
        models.append("vi-layoutxlm")
        backends.append("vi_layoutxlm")

    return SetupSelection(
        extras=_unique(extras),
        sources=_unique(sources),
        models=_unique(models),
        backends=_unique(backends),
    )


def select_all_models_setup() -> SetupSelection:
    return SetupSelection(
        extras=("paddleocr", "dbnet", "vietocr", "layoutlmv3", "vi_layoutxlm"),
        sources=("paddleocr", "dbnet"),
        models=(
            "paddleocr-detector",
            "paddleocr-recognizer",
            "dbnet",
            "dbnetpp",
            "vietocr",
            "layoutlmv3-base",
            "vi-layoutxlm",
        ),
        backends=(
            "paddleocr-detector",
            "dbnet",
            "dbnetpp",
            "paddleocr-recognizer",
            "vietocr",
            "layoutlmv3",
            "vi_layoutxlm",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a deterministic model setup plan.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--pipeline",
        nargs=3,
        metavar=("DETECTOR", "RECOGNIZER", "LAYOUT"),
    )
    selection.add_argument("--all-models", action="store_true")
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = (
            select_all_models_setup() if args.all_models else select_pipeline_setup(*args.pipeline)
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.format == "json":
        print(json.dumps(asdict(plan), indent=2))
    else:
        for field, values in asdict(plan).items():
            print(f"{field}={','.join(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
