"""Base DataModule defining the CSI sample convention.

A sample is ``(x, y)`` with ``x`` a float32 tensor of shape ``sample_shape``.
Classification targets are int64 class indices; regression targets are float32
scalars. Every DataModule exposes model-output and checkpoint metadata so the
training runner can select the correct supervised task.
"""

from pathlib import Path
from typing import Literal

import lightning as L
from torch.utils.data import DataLoader, Dataset


class CSIDataModule(L.LightningDataModule):
    name: str
    sample_shape: tuple[int, ...]
    class_names: list[str]
    task_type: Literal["classification", "regression"] = "classification"
    ordered_values: tuple[float, ...] | None = None
    target_range: tuple[float, float] | None = None
    checkpoint_monitor = "val/acc"
    checkpoint_mode: Literal["min", "max"] = "max"

    def __init__(self, batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def output_dim(self) -> int:
        return self.num_classes

    def _loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _require(path: Path, layout: str) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Expected layout:\n{layout}"
            )
