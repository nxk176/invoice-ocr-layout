"""Detector fine-tuning dispatch to revision-pinned official runtimes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from invoice_ocr.exceptions import ConfigurationError, DependencyUnavailableError


def train_detector(
    model: str,
    annotation_paths: list[Path],
    data_root: Path,
    output_dir: Path,
    resume: bool,
    device: str,
) -> None:
    if not annotation_paths:
        raise ValueError("detector training received no validated annotations")
    if model == "paddleocr":
        executable = shutil.which("python")
        config = Path("configs/models/paddleocr_detector_train.yaml")
        if executable is None or not config.is_file():
            raise ConfigurationError(
                "PaddleOCR detector training requires an official training YAML derived for "
                "the installed revision; configs/models/paddleocr_detector_train.yaml is absent."
            )
        command = [executable, "tools/train.py", "-c", str(config)]
    elif model in {"dbnet", "dbnetpp"}:
        executable = shutil.which("python")
        if executable is None or shutil.which("experiment") is None:
            raise DependencyUnavailableError(
                f"{model} training requires the official MhLiao/DB checkout on PYTHONPATH "
                "and its 'experiment' entry point."
            )
        command = [executable, "train.py", str(output_dir)]
    else:
        raise ConfigurationError(f"unsupported detector training model: {model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = {
        "device": device,
        "resume_requested": resume,
        "annotation_count": len(annotation_paths),
    }
    (output_dir / "TRAINING_RUNTIME.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in runtime.items()) + "\n",
        encoding="utf-8",
    )
    subprocess.run(command, check=True, cwd=str(data_root))
