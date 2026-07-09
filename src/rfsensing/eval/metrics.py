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