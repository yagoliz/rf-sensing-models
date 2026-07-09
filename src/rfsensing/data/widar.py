"""Widar 3.0 gesture dataset (22 gestures), SenseFi BVP packaging.

Each sample file is a text CSV of 22*20*20 floats: 22 time steps of 20x20
body-coordinate velocity profiles (BVP). Class folders are named
``<index>-<Name>`` starting at 1. No validation split exists; the val
loader serves the test set (matches SenseFi's protocol).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rfsensing.data import register
from rfsensing.data.base import CSIDataModule

_MEAN, _STD = 0.0025, 0.0119

_LAYOUT = """\
Widardata/
├── train/<idx>-<Name>/*.csv   (e.g. 1-Push&Pull)
└── test/<idx>-<Name>/*.csv"""


class _BVPFolderDataset(Dataset):
    def __init__(self, split_dir: Path, folder_names: list[str]):
        self.files: list[Path] = []
        self.labels: list[int] = []
        for idx, folder in enumerate(folder_names):
            for f in sorted((split_dir / folder).iterdir()):
                if f.is_file():
                    self.files.append(f)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = np.genfromtxt(self.files[i], delimiter=",").astype(np.float32)
        x = (x - _MEAN) / _STD
        return torch.from_numpy(x.reshape(22, 20, 20)), self.labels[i]


@register("widar")
class WidarDataModule(CSIDataModule):
    name = "widar"
    sample_shape = (22, 20, 20)

    def __init__(self, root, batch_size=64, num_workers=0):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.root = Path(root) / "Widardata"
        self._require(self.root / "train", _LAYOUT)
        self._require(self.root / "test", _LAYOUT)
        folders = [d.name for d in (self.root / "train").iterdir() if d.is_dir()]
        self.folder_names = sorted(folders, key=lambda n: int(n.split("-", 1)[0]))
        self.class_names = [n.split("-", 1)[1] for n in self.folder_names]

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_set = _BVPFolderDataset(self.root / "train", self.folder_names)
        self.test_set = _BVPFolderDataset(self.root / "test", self.folder_names)

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.test_set)

    def test_dataloader(self):
        return self._loader(self.test_set)