from __future__ import annotations

import pytest

from invoice_ocr.training.layout import layout_trainer_progress_arguments


def test_layout_training_progress_is_live_and_step_based() -> None:
    progress = layout_trainer_progress_arguments(
        training_samples=425,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
    )

    assert progress == {
        "disable_tqdm": False,
        "logging_strategy": "steps",
        "logging_steps": 3,
        "logging_first_step": True,
    }


def test_layout_training_progress_uses_at_least_one_logging_step() -> None:
    progress = layout_trainer_progress_arguments(
        training_samples=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
    )

    assert progress["logging_steps"] == 1


@pytest.mark.parametrize(
    ("training_samples", "batch_size", "gradient_accumulation_steps"),
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_layout_training_progress_rejects_non_positive_values(
    training_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        layout_trainer_progress_arguments(
            training_samples,
            batch_size,
            gradient_accumulation_steps,
        )
