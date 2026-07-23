import pytest
import torch

from rfsensing.eval.metrics import (
    confusion_matrix,
    mean_absolute_error,
    rank_k_accuracy,
    rounded_accuracy,
    tolerance_accuracy,
)


def test_rank_k_accuracy():
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0],  # target 1: correct at k=1
            [0.8, 0.15, 0.05],  # target 2: wrong at k=1 and k=2, hit at k=3
        ]
    )
    targets = torch.tensor([1, 2])
    assert rank_k_accuracy(logits, targets, k=1) == 0.5
    assert rank_k_accuracy(logits, targets, k=2) == 0.5
    assert rank_k_accuracy(logits, targets, k=3) == 1.0


def test_confusion_matrix():
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0],  # pred 1, true 1
            [0.8, 0.15, 0.05],  # pred 0, true 2
        ]
    )
    targets = torch.tensor([1, 2])
    cm = confusion_matrix(logits, targets, num_classes=3)
    expected = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert torch.equal(cm, expected)


def test_count_metrics_accept_integer_and_float_tensors():
    predictions = torch.tensor([-0.5, 2.2, 5.8])
    targets = torch.tensor([0, 3, 5])
    assert mean_absolute_error(predictions, targets) == pytest.approx(0.7)
    assert tolerance_accuracy(predictions, targets) == pytest.approx(1.0)
    assert rounded_accuracy(predictions, targets) == pytest.approx(2 / 3)


def test_count_metrics_validate_arguments():
    predictions = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="same shape"):
        mean_absolute_error(predictions, torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        tolerance_accuracy(predictions, predictions, tolerance=-1)
    with pytest.raises(ValueError, match="minimum"):
        rounded_accuracy(predictions, predictions, minimum=5, maximum=0)
