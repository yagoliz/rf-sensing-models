"""Notebook-facing experiment entry point."""

from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from rfsensing.data.base import CSIDataModule
from rfsensing.train.module import ClassificationModule


@dataclass
class Result:
    metrics: dict[str, float]
    checkpoint_path: Path
    log_dir: Path


def run(
    net: nn.Module,
    dm: CSIDataModule,
    *,
    max_epochs: int = 50,
    name: str | None = None,
    seed: int = 42,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    accelerator: str = "auto",
    runs_dir: str | Path = "runs",
) -> Result:
    """Train ``net`` on ``dm``, test with the best checkpoint, return metrics."""
    L.seed_everything(seed, workers=True)
    module = ClassificationModule(
        net, num_classes=dm.num_classes, lr=lr, weight_decay=weight_decay
    )
    experiment = name or type(net).__name__.lower()
    logger = TensorBoardLogger(str(runs_dir), name=f"{dm.name}/{experiment}")
    checkpoint = ModelCheckpoint(monitor="val/acc", mode="max", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=10,
    )
    trainer.fit(module, datamodule=dm)
    test_metrics = trainer.test(module, datamodule=dm, ckpt_path="best")[0]
    return Result(
        metrics={k: float(v) for k, v in test_metrics.items()},
        checkpoint_path=Path(checkpoint.best_model_path),
        log_dir=Path(trainer.log_dir),
    )