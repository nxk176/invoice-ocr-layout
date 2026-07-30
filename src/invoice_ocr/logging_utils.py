"""Logging configuration shared by CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_path: Path | None = None, verbose: bool = False) -> logging.Logger:
    """Configure console logging and an optional UTF-8 run log."""
    logger = logging.getLogger("invoice_ocr")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

