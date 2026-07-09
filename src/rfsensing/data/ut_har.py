"""UT-HAR: 7-activity HAR dataset (Yousefi et al.), SenseFi packaging.

The ``.csv`` files ship numpy binary content, hence ``np.load``.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from rfsensing.data import register
from rfsensing.data.base import CSIDataModule

_LAYOUT = """\
UT_HAR/
├── data/   X_train.csv X_val.csv X_test.csv   (numpy .npy binary content)
└── label/  y_train.csv y_val.csv y_test.csv"""


@register("ut_har")
class UTHARDataModule(CSIDataModule):
    name = "ut_har"
    sample_shape = (1, 250, 90)
    # Raw labels are unnamed integers 0..6; the original paper's activities
    # are lie down / fall / walk / pick up / run / sit down / stand up.
    class_names = [f"activity_{i}" for i in range(7)]

    def __init__(self, root, batch_size=64, num_workers=0):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.root = Path(root) / "UT_HAR"
        self._require(self.root / "data" / "X_train.csv", _LAYOUT)
        self._require(self.root / "label" / "y_train.csv", _LAYOUT)

    def _load_split(self, split: str) -> TensorDataset:
        with open(self.root / "data" / f"X_{split}.csv", "rb") as f:
            x = np.load(f)
        with open(self.root / "label" / f"y_{split}.csv", "rb") as f:
            y = np.load(f)
        x = x.reshape(len(x), 1, 250, 90).astype(np.float32)
        return TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_set = self._load_split("train")
            self.val_set = self._load_split("val")
        if stage in ("test", None):
            self.test_set = self._load_split("test")

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)