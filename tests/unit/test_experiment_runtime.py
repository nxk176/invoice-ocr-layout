from __future__ import annotations

from invoice_ocr.experiments.runtime import EvaluationTimer


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.unavailable_reason = None if available else "synthetic CUDA unavailable"
        self.synchronize_calls = 0
        self.reset_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def peak_memory_mb(self) -> float | None:
        return 12.5 if self.available else None


def test_cuda_synchronization_and_separate_stage_timing() -> None:
    cuda = FakeCuda(available=True)
    timer = EvaluationTimer(warmup_iterations=2, cuda=cuda)
    timer.start()
    with timer.stage("recognition"):
        pass
    with timer.document():
        pass
    result = timer.finish(1, 1, 0, 0)
    assert cuda.reset_calls == 1
    assert cuda.synchronize_calls >= 6
    assert result.recognition_time_seconds is not None
    assert result.detection_time_seconds is None
    assert result.peak_gpu_memory_mb == 12.5
    assert result.warmup_iterations == 2


def test_unavailable_resource_is_null_with_reason() -> None:
    timer = EvaluationTimer(cuda=FakeCuda(available=False))
    timer.start()
    result = timer.finish(0, 0, 0, 1)
    assert result.peak_gpu_memory_mb is None
    assert result.unavailable_reasons["peak_gpu_memory_mb"] == "synthetic CUDA unavailable"
