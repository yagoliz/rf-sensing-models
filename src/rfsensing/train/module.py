import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy


class ClassificationModule(L.LightningModule):
    """Wraps any classification net with loss, metrics and optimizer."""

    def __init__(self, net: nn.Module, num_classes: int, lr=1e-3, weight_decay=0.01):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes)
        self.val_acc = MulticlassAccuracy(num_classes)
        self.test_acc = MulticlassAccuracy(num_classes)

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, acc: MulticlassAccuracy, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc.update(logits, y)
        self.log(f"{stage}/loss", loss)
        self.log(f"{stage}/acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_acc, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.val_acc, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, self.test_acc, "test")

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