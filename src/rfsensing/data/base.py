"""Base DataModule defining the CSI sample convention.

A sample is ``(x, y)`` with ``x`` a float32 tensor of shape ``sample_shape``
and ``y`` an int64 class index. Every DataModule exposes ``name``,
``sample_shape``, ``num_classes`` and ``class_names``; models are always
built from these. A future HDF5-backed DataModule for in-house captures
implements this same contract.
"""

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, Dataset


class CSIDataModule(L.LightningDataModule):
    name: str
    sample_shape: tuple[int, ...]
    class_names: list[str]

    def __init__(self, batch_size: int = 64, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

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