import torch
from torchmetrics.functional.classification import multiclass_confusion_matrix


def rank_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Fraction of samples whose true class is within the top-k logits."""
    topk = logits.topk(k, dim=1).indices
    hits = (topk == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


def confusion_matrix(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Confusion matrix with rows = true class, columns = predicted class."""
    return multiclass_confusion_matrix(logits, targets, num_classes)


def _matching_float_tensors(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have the same shape, "
            f"got {tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    return predictions.float(), targets.float()


def mean_absolute_error(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    """Mean absolute difference between predicted and true counts."""
    predictions, targets = _matching_float_tensors(predictions, targets)
    return (predictions - targets).abs().mean().item()


def tolerance_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    tolerance: float = 1.0,
) -> float:
    """Fraction of count predictions within ``tolerance`` of the target."""
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    predictions, targets = _matching_float_tensors(predictions, targets)
    return ((predictions - targets).abs() <= tolerance).float().mean().item()


def rounded_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    minimum: int = 0,
    maximum: int = 5,
) -> float:
    """Exact accuracy after rounding and clamping scalar count predictions."""
    if minimum > maximum:
        raise ValueError(
            f"minimum must not exceed maximum, got {minimum} > {maximum}"
        )
    predictions, targets = _matching_float_tensors(predictions, targets)
    predictions = predictions.round().clamp(minimum, maximum)
    return (predictions == targets).float().mean().item()
