from __future__ import annotations

from pathlib import Path

import pytest

from invoice_ocr.experiments.training_modes import (
    BestCheckpointSelector,
    CheckpointCandidate,
    configure_layout_trainability,
)


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class FakeModel:
    def __init__(self) -> None:
        self.encoder = FakeParameter()
        self.classifier = FakeParameter()

    def named_parameters(self) -> list[tuple[str, FakeParameter]]:
        return [
            ("layoutlmv3.encoder.layer.weight", self.encoder),
            ("classifier.weight", self.classifier),
        ]


def test_linear_probe_freezes_encoder_and_trains_task_head() -> None:
    model = FakeModel()
    configure_layout_trainability(model, "linear_probe")
    assert model.encoder.requires_grad is False
    assert model.classifier.requires_grad is True


def test_full_finetune_allows_encoder_gradients() -> None:
    model = FakeModel()
    configure_layout_trainability(model, "full_finetune")
    assert model.encoder.requires_grad is True
    assert model.classifier.requires_grad is True


def test_best_checkpoint_only_consumes_validation_metrics(tmp_path: Path) -> None:
    selector = BestCheckpointSelector("entity_f1", greater_is_better=True)
    with pytest.raises(ValueError, match="only consume validation"):
        selector.add(CheckpointCandidate(tmp_path / "test", 1, 0.99, evaluated_split="test"))
    selector.add(CheckpointCandidate(tmp_path / "epoch-1", 1, 0.70, evaluated_split="validation"))
    selector.add(CheckpointCandidate(tmp_path / "epoch-2", 2, 0.80, evaluated_split="validation"))
    assert selector.best().path.name == "epoch-2"
