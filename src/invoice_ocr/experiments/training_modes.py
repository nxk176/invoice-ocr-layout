"""Layout training modes and validation-only best-checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LayoutTrainingMode = Literal["linear_probe", "full_finetune", "generic_kie_checkpoint"]


def configure_layout_trainability(model: Any, mode: LayoutTrainingMode) -> None:
    parameters = list(model.named_parameters())
    if mode == "full_finetune":
        for _, parameter in parameters:
            parameter.requires_grad = True
        return
    if mode == "generic_kie_checkpoint":
        raise ValueError("generic KIE checkpoint mode is evaluation-only")
    for _, parameter in parameters:
        parameter.requires_grad = False
    head_names = ("classifier", "classification_head", "token_classifier", "head")
    head_count = 0
    for name, parameter in parameters:
        if any(component in name.casefold() for component in head_names):
            parameter.requires_grad = True
            head_count += 1
    if head_count == 0:
        raise ValueError("cannot locate invoice task head for linear_probe mode")


@dataclass(frozen=True)
class CheckpointCandidate:
    path: Path
    epoch: float
    metric_value: float
    evaluated_split: str


class BestCheckpointSelector:
    def __init__(self, metric_name: str, greater_is_better: bool) -> None:
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.candidates: list[CheckpointCandidate] = []

    def add(self, candidate: CheckpointCandidate) -> None:
        if candidate.evaluated_split != "validation":
            raise ValueError(
                "best checkpoint selection may only consume validation metrics; "
                f"received {candidate.evaluated_split}"
            )
        self.candidates.append(candidate)

    def best(self) -> CheckpointCandidate:
        if not self.candidates:
            raise ValueError("no validation checkpoints are available for selection")
        key = (
            (lambda candidate: (-candidate.metric_value, candidate.epoch))
            if self.greater_is_better
            else (lambda candidate: (candidate.metric_value, candidate.epoch))
        )
        return min(self.candidates, key=key)
