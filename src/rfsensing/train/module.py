import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanMetric
from torchmetrics.classification import MulticlassAccuracy


class _SupervisedModule(L.LightningModule):
    def __init__(self, net: nn.Module, lr: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.trainer.max_epochs or 1, 1)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class ClassificationModule(_SupervisedModule):
    """Wrap a classifier with optional metrics for ordered class values."""

    def __init__(
        self,
        net: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        ordered_values: tuple[float, ...] | None = None,
    ):
        super().__init__(net, lr, weight_decay)
        if ordered_values is not None and len(ordered_values) != num_classes:
            raise ValueError("ordered_values length must equal num_classes")
        self.criterion = nn.CrossEntropyLoss()
        self.register_buffer(
            "ordered_values",
            torch.tensor(ordered_values or (), dtype=torch.float32),
            persistent=False,
        )
        for stage in ("train", "val", "test"):
            setattr(self, f"{stage}_acc", MulticlassAccuracy(num_classes))
            if ordered_values is not None:
                setattr(self, f"{stage}_mae", MeanAbsoluteError())
                setattr(self, f"{stage}_within_1", MeanMetric())

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = getattr(self, f"{stage}_acc")
        acc.update(logits, y)
        self.log(f"{stage}/loss", loss)
        self.log(f"{stage}/acc", acc, prog_bar=True)
        if self.ordered_values.numel():
            predictions = self.ordered_values[logits.argmax(dim=1)]
            targets = self.ordered_values[y]
            mae = getattr(self, f"{stage}_mae")
            within_1 = getattr(self, f"{stage}_within_1")
            mae.update(predictions, targets)
            within_1.update(((predictions - targets).abs() <= 1).float())
            self.log(f"{stage}/mae", mae, prog_bar=True)
            self.log(f"{stage}/within_1", within_1)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")


class RegressionModule(_SupervisedModule):
    """Wrap a scalar regressor with count-oriented evaluation metrics."""

    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        target_range: tuple[float, float] | None = None,
    ):
        super().__init__(net, lr, weight_decay)
        self.criterion = nn.MSELoss()
        self.target_range = target_range
        for stage in ("train", "val", "test"):
            setattr(self, f"{stage}_mae", MeanAbsoluteError())
            setattr(self, f"{stage}_within_1", MeanMetric())
            setattr(self, f"{stage}_rounded_acc", MeanMetric())

    def _step(self, batch, stage: str):
        x, y = batch
        predictions = self(x).reshape(-1)
        targets = y.float().reshape(-1)
        loss = self.criterion(predictions, targets)
        errors = (predictions - targets).abs()
        rounded = predictions.round()
        if self.target_range is not None:
            rounded = rounded.clamp(*self.target_range)
        mae = getattr(self, f"{stage}_mae")
        within_1 = getattr(self, f"{stage}_within_1")
        rounded_acc = getattr(self, f"{stage}_rounded_acc")
        mae.update(predictions, targets)
        within_1.update((errors <= 1).float())
        rounded_acc.update((rounded == targets).float())
        self.log(f"{stage}/loss", loss)
        self.log(f"{stage}/mae", mae, prog_bar=True)
        self.log(f"{stage}/within_1", within_1)
        self.log(f"{stage}/rounded_acc", rounded_acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")
