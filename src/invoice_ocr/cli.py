"""Command-line interface for training, inference, and benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from invoice_ocr.evaluation.benchmark import benchmark_rows, write_benchmark_reports
from invoice_ocr.exceptions import InvoiceOCRError
from invoice_ocr.logging_utils import configure_logging
from invoice_ocr.pipeline import (
    DETECTOR_NAMES,
    LAYOUT_NAMES,
    RECOGNIZER_NAMES,
    PipelineOptions,
    PipelineRunner,
    PipelineSelection,
)
from invoice_ocr.stages import (
    evaluate_prediction_directory,
    run_detect_stage,
    run_extract_stage,
    run_postprocess_stage,
    run_recognize_stage,
)
from invoice_ocr.training.datasets import validate_ground_truth
from invoice_ocr.training.trainer import (
    TrainingRequest,
    run_pipeline_training,
    run_training,
)


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--workflow-defaults",
        type=Path,
        default=Path("configs/workflow_defaults/default.yaml"),
    )
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--work-root", type=Path, default=Path("work"))


def add_pipeline_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", choices=DETECTOR_NAMES)
    parser.add_argument("--recognizer", choices=RECOGNIZER_NAMES)
    parser.add_argument("--layout", choices=LAYOUT_NAMES)
    parser.add_argument(
        "--pipeline",
        nargs=3,
        metavar=("DETECTOR", "RECOGNIZER", "LAYOUT"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m invoice_ocr.cli",
        description="Composable Vietnamese medicine invoice OCR/KIE framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run detector -> recognizer -> layout inference")
    add_common_options(run)
    add_pipeline_selection(run)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    benchmark = subparsers.add_parser("benchmark", help="benchmark pipeline combinations")
    add_common_options(benchmark)
    add_pipeline_selection(benchmark)
    benchmark.add_argument("--all-combinations", action="store_true")
    benchmark.add_argument("--input", "--data", dest="input", type=Path, required=True)
    benchmark.add_argument("--gt", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)

    detect = subparsers.add_parser("detect", help="run text detection stage")
    add_common_options(detect)
    detect.add_argument("--detector", choices=DETECTOR_NAMES, required=True)
    detect.add_argument("--input", type=Path, required=True)
    detect.add_argument("--output", type=Path, required=True)

    recognize = subparsers.add_parser("recognize", help="run recognition from stage JSONL")
    add_common_options(recognize)
    recognize.add_argument("--recognizer", choices=RECOGNIZER_NAMES, required=True)
    recognize.add_argument("--input", type=Path, required=True)
    recognize.add_argument("--output", type=Path, required=True)

    extract = subparsers.add_parser("extract", help="run layout/KIE from stage JSONL")
    add_common_options(extract)
    extract.add_argument("--layout", choices=LAYOUT_NAMES, required=True)
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)

    postprocess = subparsers.add_parser("postprocess", help="assemble canonical JSON from entities")
    add_common_options(postprocess)
    postprocess.add_argument("--input", type=Path, required=True)
    postprocess.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate predictions against GT")
    add_common_options(evaluate)
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--gt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="fine-tune one pipeline stage")
    add_common_options(train)
    train.add_argument("--stage", choices=("detector", "recognizer", "layout"), required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--gt", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("outputs/training"))

    train_pipeline = subparsers.add_parser(
        "train-pipeline", help="fine-tune detector, recognizer, then layout"
    )
    add_common_options(train_pipeline)
    train_pipeline.add_argument(
        "--pipeline",
        nargs=3,
        required=True,
        metavar=("DETECTOR", "RECOGNIZER", "LAYOUT"),
    )
    train_pipeline.add_argument("--data", type=Path, required=True)
    train_pipeline.add_argument("--gt", type=Path, required=True)
    train_pipeline.add_argument("--output", type=Path, default=Path("outputs/training"))

    validate_gt = subparsers.add_parser(
        "validate-gt", help="validate available ground-truth levels"
    )
    add_common_options(validate_gt)
    validate_gt.add_argument("--gt", type=Path, required=True)
    validate_gt.add_argument("--output", type=Path)
    return parser


def selection_from_args(args: argparse.Namespace) -> PipelineSelection:
    pipeline = args.pipeline
    explicit = (args.detector, args.recognizer, args.layout)
    if pipeline is not None and any(value is not None for value in explicit):
        raise InvoiceOCRError(
            "use either --pipeline DETECTOR RECOGNIZER LAYOUT or the three explicit flags"
        )
    if pipeline is not None:
        detector, recognizer, layout = pipeline
    elif all(value is not None for value in explicit):
        detector, recognizer, layout = explicit
    else:
        raise InvoiceOCRError(
            "pipeline selection requires --detector A --recognizer B --layout C or --pipeline A B C"
        )
    return PipelineSelection(str(detector), str(recognizer), str(layout))


def options_from_args(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        input_path=args.input,
        output_path=args.output,
        work_root=args.work_root,
        model_root=args.model_root,
        workflow_defaults=args.workflow_defaults,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        force=args.force,
        fail_fast=args.fail_fast,
        keep_intermediate=args.keep_intermediate,
        config=args.config,
    )


def _run_benchmark(args: argparse.Namespace) -> None:
    from invoice_ocr.io.paths import discover_documents

    discover_documents(args.input)
    if args.all_combinations:
        rows = benchmark_rows(True)
    else:
        selection = selection_from_args(args)
        rows = [
            {
                "detector": selection.detector,
                "recognizer": selection.recognizer,
                "layout": selection.layout,
                "status": "not_run",
            }
        ]
    for row in rows:
        selection = PipelineSelection(
            str(row["detector"]), str(row["recognizer"]), str(row["layout"])
        )
        run_output = args.output / (
            f"{selection.detector}__{selection.recognizer}__{selection.layout}"
        )
        options = options_from_args(args)
        options.output_path = run_output
        try:
            manifest = PipelineRunner(selection, options).run()
            row["status"] = manifest.status.value
            row["failed_document_count"] = str(manifest.failed_document_count)
        except Exception as exc:
            row["status"] = "failed"
            row["reason"] = str(exc)
            if args.fail_fast:
                raise
    write_benchmark_reports(args.output, rows, args.gt)


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "run":
        logger = configure_logging(args.output / "logs" / "run.log")
        PipelineRunner(selection_from_args(args), options_from_args(args), logger=logger).run()
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "detect":
        run_detect_stage(args.detector, args.input, args.output, args.model_root, args.device)
    elif args.command == "recognize":
        run_recognize_stage(args.recognizer, args.input, args.output, args.model_root, args.device)
    elif args.command == "extract":
        run_extract_stage(args.layout, args.input, args.output, args.model_root, args.device)
    elif args.command == "postprocess":
        run_postprocess_stage(args.input, args.output, args.workflow_defaults)
    elif args.command == "evaluate":
        evaluate_prediction_directory(args.input, args.gt, args.output)
    elif args.command == "validate-gt":
        report = validate_ground_truth(args.gt)
        text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        if not report.is_valid:
            raise InvoiceOCRError("ground-truth validation failed")
    elif args.command == "train":
        run_training(
            TrainingRequest(
                stage=args.stage,
                model=args.model,
                data_root=args.data,
                gt_root=args.gt,
                model_root=args.model_root,
                output_root=args.output,
                device=args.device,
                seed=args.seed,
                resume=args.resume,
                force=args.force,
            )
        )
    elif args.command == "train-pipeline":
        detector, recognizer, layout = args.pipeline
        run_pipeline_training(
            detector,
            recognizer,
            layout,
            args.data,
            args.gt,
            args.model_root,
            args.output,
            args.device,
            args.seed,
            args.resume,
            args.force,
        )
    else:
        raise InvoiceOCRError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dispatch(args)
        return 0
    except (InvoiceOCRError, FileNotFoundError, ValueError) as exc:
        configure_logging().error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
