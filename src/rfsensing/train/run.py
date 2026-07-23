"""Notebook-facing experiment entry point."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from rfsensing.data.base import CSIDataModule
from rfsensing.train.module import ClassificationModule, RegressionModule


@dataclass
class Result:
    metrics: dict[str, float]
    checkpoint_path: Path
    log_dir: Path


def _build_module(
    net: nn.Module,
    dm: CSIDataModule,
    lr: float,
    weight_decay: float,
) -> ClassificationModule | RegressionModule:
    if dm.task_type == "classification":
        return ClassificationModule(
            net,
            num_classes=dm.output_dim,
            lr=lr,
            weight_decay=weight_decay,
            ordered_values=dm.ordered_values,
        )
    if dm.task_type == "regression":
        if dm.output_dim != 1:
            raise ValueError(
                f"scalar regression requires output_dim=1, got {dm.output_dim}"
            )
        return RegressionModule(
            net,
            lr=lr,
            weight_decay=weight_decay,
            target_range=dm.target_range,
        )
    raise ValueError(f"Unsupported task_type {dm.task_type!r}")


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
    monitor: str | None = None,
    monitor_mode: Literal["min", "max"] | None = None,
) -> Result:
    """Train ``net`` on ``dm``, test with the best checkpoint, return metrics."""
    L.seed_everything(seed, workers=True)
    module = _build_module(net, dm, lr=lr, weight_decay=weight_decay)
    experiment = name or type(net).__name__.lower()
    logger = TensorBoardLogger(str(runs_dir), name=f"{dm.name}/{experiment}")
    monitor = monitor or dm.checkpoint_monitor
    monitor_mode = monitor_mode or dm.checkpoint_mode
    checkpoint = ModelCheckpoint(
        monitor=monitor, mode=monitor_mode, save_top_k=1
    )
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


def load_best_net(net: nn.Module, result: Result) -> nn.Module:
    """Load ``result``'s best-checkpoint weights into ``net`` (in place).

    The in-memory net after ``run()`` holds last-epoch weights; the best by the
    monitored validation metric live under Lightning's ``net.`` prefix.
    """
    state = torch.load(
        result.checkpoint_path, map_location="cpu", weights_only=True
    )["state_dict"]
    net.load_state_dict(
        {k.removeprefix("net."): v for k, v in state.items() if k.startswith("net.")}
    )
    return net
