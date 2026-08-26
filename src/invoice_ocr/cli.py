"""Command-line interface for training, inference, benchmarking, and experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from invoice_ocr.evaluation.benchmark import benchmark_rows, write_benchmark_reports
from invoice_ocr.exceptions import InvoiceOCRError
from invoice_ocr.io.paths import ensure_runtime_directories
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


def add_locked_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-source",
        choices=("pretrained",),
        default="pretrained",
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--layout-training-mode",
        choices=("linear_probe", "full_finetune"),
        default="full_finetune",
    )
    parser.add_argument("--selection-metric")
    parser.add_argument(
        "--maximize-metric",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--mixed-precision-mode",
        choices=("none", "fp16", "bf16"),
        default="none",
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
    benchmark.add_argument("--gt-prefix")
    benchmark.add_argument("--target-manifest", type=Path)
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
    evaluate.add_argument("--data", type=Path)
    evaluate.add_argument("--gt-prefix")
    evaluate.add_argument("--target-manifest", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)

    build_layout_gt = subparsers.add_parser(
        "build-layout-gt",
        help="align pretrained OCR text/boxes to canonical final field-value GT",
    )
    add_common_options(build_layout_gt)
    build_layout_gt.add_argument("--input", type=Path, required=True)
    build_layout_gt.add_argument("--gt", type=Path, required=True)
    build_layout_gt.add_argument("--detector", choices=DETECTOR_NAMES, required=True)
    build_layout_gt.add_argument("--recognizer", choices=RECOGNIZER_NAMES, required=True)
    build_layout_gt.add_argument("--output", type=Path, required=True)
    build_layout_gt.add_argument("--gt-prefix")
    build_layout_gt.add_argument("--target-manifest", type=Path)
    build_layout_gt.add_argument("--max-alignment-boxes", type=int, default=12)

    inspect_layout_gt = subparsers.add_parser(
        "inspect-layout-gt",
        help="validate pseudo LayoutLM annotations and print alignment coverage",
    )
    inspect_layout_gt.add_argument("--layout-gt", type=Path, required=True)

    train = subparsers.add_parser("train", help="fine-tune one pipeline stage")
    add_common_options(train)
    add_locked_training_options(train)
    train.add_argument("--stage", choices=("detector", "recognizer", "layout"), required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--data", type=Path)
    train.add_argument("--gt", type=Path)
    train.add_argument("--layout-gt", type=Path)
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

    create_split = subparsers.add_parser(
        "create-split",
        help="create an immutable deterministic train/validation/test split",
    )
    add_common_options(create_split)
    create_split.add_argument("--data", type=Path, required=True)
    create_split.add_argument("--gt", type=Path, required=True)
    create_split.add_argument("--gt-prefix")
    create_split.add_argument("--target-manifest", type=Path)
    create_split.add_argument("--output", type=Path, required=True)

    evaluate_model_parser = subparsers.add_parser(
        "evaluate-model",
        help="evaluate one pretrained or fine-tuned model on a locked split",
    )
    add_common_options(evaluate_model_parser)
    evaluate_model_parser.add_argument(
        "--stage", choices=("detector", "recognizer", "layout"), required=True
    )
    evaluate_model_parser.add_argument("--model", required=True)
    evaluate_model_parser.add_argument(
        "--checkpoint-source",
        choices=("pretrained", "linear_probe", "finetuned", "generic_kie_checkpoint"),
        default="pretrained",
    )
    evaluate_model_parser.add_argument("--checkpoint", type=Path)
    evaluate_model_parser.add_argument("--data", type=Path, required=True)
    evaluate_model_parser.add_argument("--gt", type=Path, required=True)
    evaluate_model_parser.add_argument("--layout-gt", type=Path)
    evaluate_model_parser.add_argument("--split-manifest", type=Path, required=True)
    evaluate_model_parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    evaluate_model_parser.add_argument("--output", type=Path, required=True)
    evaluate_model_parser.add_argument(
        "--layout-training-mode",
        choices=("linear_probe", "full_finetune", "generic_kie_checkpoint"),
    )
    evaluate_model_parser.add_argument("--warmup-iterations", type=int, default=0)
    evaluate_model_parser.add_argument("--validation-tolerance", default="0.01")

    compare = subparsers.add_parser(
        "compare-runs",
        help="compare locked pretrained and fine-tuned evaluation artifacts",
    )
    add_common_options(compare)
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--allow-incomparable-runs", action="store_true")

    experiment = subparsers.add_parser(
        "experiment",
        help="run the locked pretrained-versus-fine-tuned protocol",
    )
    add_common_options(experiment)
    add_pipeline_selection(experiment)
    experiment.add_argument("--all-combinations", action="store_true")
    experiment.add_argument(
        "--protocol",
        choices=("pretrained-vs-finetuned",),
        default="pretrained-vs-finetuned",
    )
    experiment.add_argument(
        "--layout-baseline-mode",
        choices=("linear_probe", "generic_kie_checkpoint"),
        default="linear_probe",
    )
    experiment.add_argument(
        "--layout-finetuned-mode",
        choices=("full_finetune",),
        default="full_finetune",
    )
    experiment.add_argument("--data", type=Path, required=True)
    experiment.add_argument("--gt", type=Path, required=True)
    experiment.add_argument("--gt-prefix")
    experiment.add_argument("--target-manifest", type=Path)
    experiment.add_argument("--layout-gt", type=Path)
    experiment.add_argument("--layout-checkpoint", type=Path)
    experiment.add_argument(
        "--layout-checkpoint-mode",
        choices=("linear_probe", "full_finetune"),
        default="full_finetune",
    )
    experiment.add_argument("--split", choices=("test",), default="test")
    experiment.add_argument("--split-manifest", type=Path)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--warmup-iterations", type=int, default=0)
    experiment.add_argument("--epochs", type=int, default=3)
    experiment.add_argument("--learning-rate", type=float, default=5e-5)
    experiment.add_argument("--gradient-accumulation-steps", type=int, default=1)
    experiment.add_argument(
        "--mixed-precision-mode",
        choices=("none", "fp16", "bf16"),
        default="none",
    )
    experiment.add_argument("--validation-tolerance", default="0.01")

    verify_models = subparsers.add_parser(
        "verify-models",
        help="verify dependency, source, checkpoint, and implementation readiness",
    )
    verify_selection = verify_models.add_mutually_exclusive_group(required=True)
    verify_selection.add_argument("--all", action="store_true")
    verify_selection.add_argument(
        "--backend",
        action="append",
        choices=(
            "paddleocr-detector",
            "dbnet",
            "dbnetpp",
            "paddleocr-recognizer",
            "vietocr",
            "layoutlmv3",
            "vi_layoutxlm",
        ),
    )
    verify_models.add_argument("--model-root", type=Path, default=Path("models"))
    verify_models.add_argument("--external-root", type=Path, default=Path("external"))
    verify_models.add_argument(
        "--require",
        choices=("any", "inference", "training", "both"),
        default="any",
    )
    verify_models.add_argument("--json", action="store_true", dest="json_output")
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
    write_benchmark_reports(
        args.output,
        rows,
        args.gt,
        data_root=args.input,
        gt_prefix=args.gt_prefix,
        target_manifest=args.target_manifest,
    )


def _dispatch_locked_training(args: argparse.Namespace) -> None:
    from invoice_ocr.experiments.train_model import (
        LockedTrainingRequest,
        train_model_locked,
    )

    data_root = args.data
    gt_root = args.gt
    if args.layout_gt is not None:
        if args.stage != "layout":
            raise InvoiceOCRError("--layout-gt is only valid with --stage layout")
        manifest_path = args.layout_gt / "manifest.json"
        if not manifest_path.is_file():
            raise InvoiceOCRError(
                f"pseudo-layout manifest not found: {manifest_path}; run build-layout-gt first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise InvoiceOCRError(f"invalid pseudo-layout manifest: {manifest_path}")
        data_root = data_root or Path(str(manifest["input_root"]))
        gt_root = gt_root or Path(str(manifest["gt_root"]))
    if data_root is None or gt_root is None:
        raise InvoiceOCRError(
            "locked training requires --data and --gt, or --layout-gt whose manifest records both"
        )
    train_model_locked(
        LockedTrainingRequest(
            stage=args.stage,
            model=args.model,
            checkpoint_source=args.checkpoint_source,
            data_root=data_root,
            gt_root=gt_root,
            layout_gt_root=args.layout_gt,
            split_manifest=args.split_manifest,
            output_dir=args.output,
            model_root=args.model_root,
            device=args.device,
            seed=args.seed,
            resume=args.resume,
            force=args.force,
            layout_training_mode=args.layout_training_mode,
            selection_metric=args.selection_metric,
            maximize_metric=args.maximize_metric,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision_mode=args.mixed_precision_mode,
            num_workers=args.num_workers,
        )
    )


def _dispatch_experiment(args: argparse.Namespace) -> int:
    from invoice_ocr.experiments.orchestrator import (
        ExperimentRequest,
        run_experiment,
    )
    from invoice_ocr.experiments.split import create_locked_split

    if args.all_combinations and (
        args.pipeline is not None
        or args.detector is not None
        or args.recognizer is not None
        or args.layout is not None
    ):
        raise InvoiceOCRError("use either --all-combinations or one pipeline selection, not both")
    pipeline = None if args.all_combinations else selection_from_args(args)
    split_manifest = args.split_manifest or args.gt / "splits" / "split_v1.json"
    if not split_manifest.is_file():
        create_locked_split(
            args.data,
            args.gt,
            split_manifest,
            args.seed,
            args.force,
            args.gt_prefix,
            args.target_manifest,
        )
    if args.layout_checkpoint is not None:
        if pipeline is None:
            raise InvoiceOCRError("--layout-checkpoint requires one explicit --pipeline")
        from invoice_ocr.experiments.evaluate_pipeline import (
            PipelineEvaluationRequest,
            evaluate_pipeline,
        )

        result = evaluate_pipeline(
            PipelineEvaluationRequest(
                pipeline=pipeline,
                checkpoints={
                    "detector": None,
                    "recognizer": None,
                    "layout": args.layout_checkpoint,
                },
                run_kind="finetuned",
                data_root=args.data,
                gt_root=args.gt,
                split_manifest=split_manifest,
                output_dir=args.output,
                model_root=args.model_root,
                work_root=args.work_root / args.output.name,
                workflow_defaults=args.workflow_defaults,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                warmup_iterations=args.warmup_iterations,
                seed=args.seed,
                resume=args.resume,
                force=args.force,
                validation_tolerance=args.validation_tolerance,
                baseline_mode=(
                    "linear_probe" if args.layout_checkpoint_mode == "linear_probe" else None
                ),
                finetuned_mode=(
                    "full_finetune" if args.layout_checkpoint_mode == "full_finetune" else None
                ),
                gt_prefix=args.gt_prefix,
                target_manifest=args.target_manifest,
            )
        )
        manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
        return 0 if isinstance(manifest, dict) and manifest.get("status") == "success" else 1
    outcome = run_experiment(
        ExperimentRequest(
            output_dir=args.output,
            data_root=args.data,
            gt_root=args.gt,
            split_manifest=split_manifest,
            layout_gt_root=args.layout_gt,
            gt_prefix=args.gt_prefix,
            target_manifest=args.target_manifest,
            pipeline=pipeline,
            all_combinations=args.all_combinations,
            protocol=args.protocol,
            layout_baseline_mode=args.layout_baseline_mode,
            layout_finetuned_mode=args.layout_finetuned_mode,
            model_root=args.model_root,
            work_root=args.work_root,
            workflow_defaults=args.workflow_defaults,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup_iterations=args.warmup_iterations,
            seed=args.seed,
            resume=args.resume,
            force=args.force,
            fail_fast=args.fail_fast,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision_mode=args.mixed_precision_mode,
            validation_tolerance=args.validation_tolerance,
        )
    )
    return 1 if outcome.all_failed else 0


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "verify-models":
        from invoice_ocr.model_verification import run_verification

        return run_verification(args)
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
        evaluate_prediction_directory(
            args.input,
            args.gt,
            args.output,
            data_root=args.data,
            gt_prefix=args.gt_prefix,
            target_manifest=args.target_manifest,
        )
    elif args.command == "build-layout-gt":
        from invoice_ocr.layout_gt.builder import (
            LayoutGTBuildRequest,
            build_layout_ground_truth,
        )

        build_layout_ground_truth(
            LayoutGTBuildRequest(
                input_root=args.input,
                gt_root=args.gt,
                output_dir=args.output,
                detector_name=args.detector,
                recognizer_name=args.recognizer,
                model_root=args.model_root,
                device=args.device,
                gt_prefix=args.gt_prefix,
                target_manifest=args.target_manifest,
                force=args.force,
                max_alignment_boxes=args.max_alignment_boxes,
            )
        )
    elif args.command == "inspect-layout-gt":
        from invoice_ocr.layout_gt.builder import inspect_layout_ground_truth

        print(
            json.dumps(
                inspect_layout_ground_truth(args.layout_gt),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "validate-gt":
        report = validate_ground_truth(args.gt)
        text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        if not report.is_valid:
            raise InvoiceOCRError("ground-truth validation failed")
    elif args.command == "create-split":
        from invoice_ocr.experiments.split import create_locked_split

        manifest = create_locked_split(
            args.data,
            args.gt,
            args.output,
            args.seed,
            args.force,
            args.gt_prefix,
            args.target_manifest,
        )
        print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif args.command == "evaluate-model":
        from invoice_ocr.experiments.evaluate_model import (
            ModelEvaluationRequest,
            evaluate_model,
        )

        evaluate_model(
            ModelEvaluationRequest(
                stage=args.stage,
                model=args.model,
                checkpoint_source=args.checkpoint_source,
                checkpoint=args.checkpoint,
                layout_gt_root=args.layout_gt,
                data_root=args.data,
                gt_root=args.gt,
                split_manifest=args.split_manifest,
                split=args.split,
                output_dir=args.output,
                model_root=args.model_root,
                work_root=args.work_root,
                workflow_defaults=args.workflow_defaults,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                warmup_iterations=args.warmup_iterations,
                seed=args.seed,
                resume=args.resume,
                force=args.force,
                baseline_mode=(
                    args.layout_training_mode
                    if args.stage == "layout" and args.checkpoint_source != "finetuned"
                    else None
                ),
                finetuned_mode=(
                    args.layout_training_mode
                    if args.stage == "layout" and args.checkpoint_source == "finetuned"
                    else None
                ),
                validation_tolerance=args.validation_tolerance,
            )
        )
    elif args.command == "compare-runs":
        from invoice_ocr.experiments.comparison import compare_runs

        compare_runs(
            args.before,
            args.after,
            args.output,
            allow_incomparable_runs=args.allow_incomparable_runs,
        )
    elif args.command == "experiment":
        return _dispatch_experiment(args)
    elif args.command == "train":
        if args.split_manifest is not None:
            _dispatch_locked_training(args)
        else:
            if args.layout_gt is not None:
                raise InvoiceOCRError("--layout-gt training requires --split-manifest")
            if args.data is None or args.gt is None:
                raise InvoiceOCRError("training requires --data and --gt")
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
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ensure_runtime_directories(Path.cwd())
        return dispatch(args)
    except (InvoiceOCRError, FileNotFoundError, ValueError) as exc:
        configure_logging().error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
