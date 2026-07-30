"""Joint classification + batch-hard triplet training for person Re-ID.

The encoder is any registry model exposing ``embed(x)`` and ``head``: one
embedding pass feeds both the auxiliary identity classifier (raw embeddings)
and the cosine triplet loss (L2-normalized embeddings). The classifier
stabilizes training but is never used for gallery matching.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from rfsensing.data.reid import IdentityBatchSampler
from rfsensing.eval.reid import retrieval_metrics, score_gallery_probe
from rfsensing.train.module import _SupervisedModule


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.3,
) -> torch.Tensor:
    """Cosine batch-hard triplet loss over one P×K batch.

    Each anchor pairs its farthest positive against its closest negative:
    ``relu(d_pos - d_neg + margin)`` averaged over anchors.
    """
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    labels = labels.reshape(-1)
    z = F.normalize(embeddings.float(), dim=1)
    distances = 1.0 - z @ z.T
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = same & ~eye
    negative_mask = ~same
    if not positive_mask.any(dim=1).all():
        raise ValueError("every anchor needs at least one positive in the batch")
    if not negative_mask.any(dim=1).all():
        raise ValueError("every anchor needs at least one negative in the batch")
    hardest_positive = distances.masked_fill(~positive_mask, -torch.inf).amax(1)
    hardest_negative = distances.masked_fill(~negative_mask, torch.inf).amin(1)
    return F.relu(hardest_positive - hardest_negative + margin).mean()


class ReIDModule(_SupervisedModule):
    """Lightning module for open-set Re-ID embedding training.

    Validation expects two loaders — gallery first, known probes second (the
    Re-ID DataModule contract) — and logs retrieval ``val/mAP``, ``val/rank1``
    and ``val/rank3`` (rank capped at the enrolled identity count).
    """

    def __init__(
        self,
        net: nn.Module,
        num_train_identities: int,
        *,
        triplet_margin: float = 0.3,
        triplet_weight: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        if not callable(getattr(net, "embed", None)):
            raise ValueError("net must expose a callable embed(x) method")
        head = getattr(net, "head", None)
        out_features = getattr(head, "out_features", None)
        if head is None or out_features is None:
            raise ValueError("net must expose a head classifier layer")
        if out_features != num_train_identities:
            raise ValueError(
                f"classifier width {out_features} does not match "
                f"{num_train_identities} training identities"
            )
        if triplet_weight < 0:
            raise ValueError(f"triplet_weight must be >= 0, got {triplet_weight}")
        if triplet_margin <= 0:
            raise ValueError(f"triplet_margin must be > 0, got {triplet_margin}")
        super().__init__(net, lr, weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self._val_embeddings: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        self._val_labels: dict[int, list[torch.Tensor]] = {0: [], 1: []}

    def training_step(self, batch, batch_idx):
        x, y = batch
        raw = self.net.embed(x)
        logits = self.net.head(raw)
        ce_loss = self.criterion(logits, y)
        triplet_loss = batch_hard_triplet_loss(
            F.normalize(raw, dim=1), y, margin=self.hparams.triplet_margin
        )
        loss = ce_loss + self.hparams.triplet_weight * triplet_loss
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/ce_loss", ce_loss)
        self.log("train/triplet_loss", triplet_loss)
        return loss

    def on_train_epoch_start(self):
        loader = self.trainer.train_dataloader
        sampler = getattr(loader, "batch_sampler", None)
        if isinstance(sampler, IdentityBatchSampler):
            sampler.set_epoch(self.current_epoch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        z = F.normalize(self.net.embed(x), dim=1)
        self._val_embeddings[dataloader_idx].append(z)
        self._val_labels[dataloader_idx].append(y)

    def _clear_validation_buffers(self):
        for store in (self._val_embeddings, self._val_labels):
            for buffer in store.values():
                buffer.clear()

    def on_validation_epoch_end(self):
        sanity = self._trainer is not None and self.trainer.sanity_checking
        if sanity or not self._val_embeddings[0] or not self._val_embeddings[1]:
            self._clear_validation_buffers()
            return
        scores = score_gallery_probe(
            torch.cat(self._val_embeddings[0]),
            torch.cat(self._val_labels[0]),
            torch.cat(self._val_embeddings[1]),
        )
        probe_labels = torch.cat(self._val_labels[1])
        num_identities = scores.identity_labels.numel()
        capped = min(3, num_identities)
        metrics = retrieval_metrics(
            scores,
            probe_labels,
            torch.ones(probe_labels.numel(), dtype=torch.bool),
            ranks=(1, capped) if capped > 1 else (1,),
        )
        self.log("val/mAP", metrics["mAP"], prog_bar=True)
        self.log("val/rank1", metrics["rank1"])
        # With fewer than 3 enrolled identities rank-3 saturates at the
        # capped rank, which is the identical quantity.
        self.log("val/rank3", metrics[f"rank{capped}"])
        self._clear_validation_buffers()
