"""Synchronized wall-time and peak resource measurement."""

from __future__ import annotations

import contextlib
import statistics
import threading
import time
from collections.abc import Iterator
from typing import Any, ClassVar, Protocol

import psutil

from invoice_ocr.experiments.contracts import EvaluationTiming


class CudaHooks(Protocol):
    available: bool
    unavailable_reason: str | None

    def synchronize(self) -> None: ...

    def reset_peak_memory_stats(self) -> None: ...

    def peak_memory_mb(self) -> float | None: ...


class TorchCudaHooks:
    def __init__(self) -> None:
        self._torch: Any | None = None
        self.available = False
        self.unavailable_reason: str | None = "CUDA is unavailable"
        try:
            import torch

            self._torch = torch
            self.available = bool(torch.cuda.is_available())
            self.unavailable_reason = (
                None if self.available else "torch.cuda.is_available() is false"
            )
        except ImportError:
            self.unavailable_reason = "PyTorch is not installed"

    def synchronize(self) -> None:
        if self.available and self._torch is not None:
            self._torch.cuda.synchronize()

    def reset_peak_memory_stats(self) -> None:
        if self.available and self._torch is not None:
            self._torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self) -> float | None:
        if not self.available or self._torch is None:
            return None
        return float(self._torch.cuda.max_memory_allocated()) / (1024 * 1024)


class PeakCpuMonitor:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        process = psutil.Process()
        self.peak_bytes = process.memory_info().rss

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
                except psutil.Error:
                    return

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        return self.peak_bytes / (1024 * 1024)


class EvaluationTimer:
    STAGES: ClassVar[dict[str, str]] = {
        "model_load": "model_load_time_seconds",
        "preprocessing": "preprocessing_time_seconds",
        "detection": "detection_time_seconds",
        "recognition": "recognition_time_seconds",
        "layout_inference": "layout_inference_time_seconds",
        "table_reconstruction": "table_reconstruction_time_seconds",
        "postprocessing": "postprocessing_time_seconds",
        "validation": "validation_time_seconds",
        "evaluation": "evaluation_time_seconds",
    }

    def __init__(
        self,
        warmup_iterations: int = 0,
        cuda: CudaHooks | None = None,
    ) -> None:
        if warmup_iterations < 0:
            raise ValueError("warmup iterations must be non-negative")
        self.warmup_iterations = warmup_iterations
        self.cuda = cuda or TorchCudaHooks()
        self.cpu = PeakCpuMonitor()
        self.stage_seconds: dict[str, float] = {}
        self.document_seconds: list[float] = []
        self.start_time = 0.0
        self.peak_cpu_mb: float | None = None
        self.peak_gpu_mb: float | None = None

    def start(self) -> None:
        self.cuda.synchronize()
        self.cuda.reset_peak_memory_stats()
        self.cpu.start()
        self.start_time = time.perf_counter()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in self.STAGES:
            raise ValueError(f"unknown timed stage: {name}")
        self.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.cuda.synchronize()
            self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + (
                time.perf_counter() - started
            )

    @contextlib.contextmanager
    def document(self) -> Iterator[None]:
        self.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.cuda.synchronize()
            self.document_seconds.append(time.perf_counter() - started)

    def finish(
        self,
        processed_documents: int,
        processed_pages: int,
        failed_documents: int,
        skipped_documents: int,
    ) -> EvaluationTiming:
        self.cuda.synchronize()
        total = max(0.0, time.perf_counter() - self.start_time)
        self.peak_cpu_mb = self.cpu.stop()
        self.peak_gpu_mb = self.cuda.peak_memory_mb()
        unavailable: dict[str, str] = {}
        values: dict[str, float | None] = {}
        for stage, field in self.STAGES.items():
            values[field] = self.stage_seconds.get(stage)
            if values[field] is None:
                unavailable[field] = f"{stage} is not applicable to this evaluation"
        if self.peak_gpu_mb is None:
            unavailable["peak_gpu_memory_mb"] = (
                self.cuda.unavailable_reason or "GPU peak memory is unavailable"
            )
        mean_document = statistics.fmean(self.document_seconds) if self.document_seconds else None
        median_document = (
            statistics.median(self.document_seconds) if self.document_seconds else None
        )
        return EvaluationTiming(
            total_wall_time_seconds=total,
            **values,
            mean_time_per_document_seconds=mean_document,
            median_time_per_document_seconds=median_document,
            mean_time_per_page_seconds=(total / processed_pages if processed_pages else None),
            throughput_documents_per_second=(
                processed_documents / total if processed_documents and total > 0 else None
            ),
            throughput_pages_per_second=(
                processed_pages / total if processed_pages and total > 0 else None
            ),
            peak_cpu_ram_mb=self.peak_cpu_mb,
            peak_gpu_memory_mb=self.peak_gpu_mb,
            processed_document_count=processed_documents,
            processed_page_count=processed_pages,
            failed_document_count=failed_documents,
            skipped_document_count=skipped_documents,
            warmup_iterations=self.warmup_iterations,
            unavailable_reasons=unavailable,
        )
