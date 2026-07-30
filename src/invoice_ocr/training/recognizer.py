"""Recognizer fine-tuning dispatch."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from invoice_ocr.exceptions import ConfigurationError, DependencyUnavailableError


def train_recognizer(
    model: str,
    annotation_paths: list[Path],
    data_root: Path,
    output_dir: Path,
    resume: bool,
    device: str,
) -> None:
    if not annotation_paths:
        raise ValueError("recognizer training received no validated transcriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    if model == "vietocr":
        if shutil.which("vietocr") is None:
            raise DependencyUnavailableError(
                "VietOCR training requires its official CLI. Install the pinned vietocr "
                "package and prepare a dataset config from validated recognition annotations."
            )
        config = Path("configs/models/vietocr_train.yaml")
        if not config.is_file():
            raise ConfigurationError(
                "configs/models/vietocr_train.yaml must be reviewed with real dataset paths "
                "before VietOCR training; the framework does not invent them."
            )
        command = ["vietocr", "train", "--config", str(config)]
    elif model == "paddleocr":
        if shutil.which("python") is None:
            raise DependencyUnavailableError("Python executable not found")
        config = Path("configs/models/paddleocr_recognizer_train.yaml")
        if not config.is_file():
            raise ConfigurationError(
                "configs/models/paddleocr_recognizer_train.yaml is required for the installed "
                "official PaddleOCR revision."
            )
        command = ["python", "tools/train.py", "-c", str(config)]
    else:
        raise ConfigurationError(f"unsupported recognizer training model: {model}")
    subprocess.run(command, check=True, cwd=str(data_root))
