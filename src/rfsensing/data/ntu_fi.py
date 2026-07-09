"""NTU-Fi HAR (6 activities) and HumanID (14 identities), SenseFi packaging.

Each sample is a ``.mat`` file whose ``CSIamp`` entry is (342, 2000):
3 antennas x 114 subcarriers, 2000 packets. Following SenseFi, packets are
downsampled by 4 to 500 and amplitudes normalized with fixed constants.
No validation split exists; the val loader serves the test set (this matches
SenseFi's protocol — model selection and testing are not independent).
"""

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from rfsensing.data import register
from rfsensing.data.base import CSIDataModule

_MEAN, _STD = 42.3199, 4.9802


class _MatFolderDataset(Dataset):
    def __init__(self, split_dir: Path, class_names: list[str]):
        self.files: list[Path] = []
        self.labels: list[int] = []
        for idx, cls in enumerate(class_names):
            for f in sorted((split_dir / cls).glob("*.mat")):
                self.files.append(f)
                self.labels.append(idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = sio.loadmat(self.files[i])["CSIamp"].astype(np.float32)
        x = (x - _MEAN) / _STD
        x = x[:, ::4][:, :500]
        x = x.reshape(3, 114, 500)
        return torch.from_numpy(x), self.labels[i]


class _NTUFiBase(CSIDataModule):
    sample_shape = (3, 114, 500)
    dirname: str  # set by subclasses

    def __init__(self, root, batch_size=64, num_workers=0):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.root = Path(root) / self.dirname
        layout = (
            f"{self.dirname}/\n"
            "├── train_amp/<class name>/*.mat\n"
            "└── test_amp/<class name>/*.mat"
        )
        self._require(self.root / "train_amp", layout)
        self._require(self.root / "test_amp", layout)
        self.class_names = sorted(
            d.name for d in (self.root / "train_amp").iterdir() if d.is_dir()
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_set = _MatFolderDataset(
                self.root / "train_amp", self.class_names
            )
        self.test_set = _MatFolderDataset(self.root / "test_amp", self.class_names)

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.test_set)

    def test_dataloader(self):
        return self._loader(self.test_set)


@register("ntu_fi_har")
class NTUFiHARDataModule(_NTUFiBase):
    name = "ntu_fi_har"
    dirname = "NTU-Fi_HAR"


@register("ntu_fi_humanid")
class NTUFiHumanIDDataModule(_NTUFiBase):
    name = "ntu_fi_humanid"
    dirname = "NTU-Fi-HumanID"