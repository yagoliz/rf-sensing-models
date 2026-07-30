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
from torch.utils.data import DataLoader, Dataset

from rfsensing.data import register
from rfsensing.data.base import CSIDataModule
from rfsensing.data.reid import IdentityBatchSampler, make_identity_split

_MEAN, _STD = 42.3199, 4.9802


def _mat_records(
    split_dir: Path, class_names: list[str], label_by_class: dict[str, int] | None = None
) -> tuple[list[Path], list[int]]:
    """Collect (file, label) pairs; labels default to the class-list index."""
    files: list[Path] = []
    labels: list[int] = []
    for idx, cls in enumerate(class_names):
        label = idx if label_by_class is None else label_by_class[cls]
        for f in sorted((split_dir / cls).glob("*.mat")):
            files.append(f)
            labels.append(label)
    return files, labels


class _MatFolderDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        class_names: list[str],
        label_by_class: dict[str, int] | None = None,
    ):
        self.files, self.labels = _mat_records(
            split_dir, class_names, label_by_class
        )

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


@register("ntu_fi_humanid_reid")
class NTUFiHumanIDReIDDataModule(CSIDataModule):
    """Identity-disjoint open-set Re-ID view of NTU-Fi HumanID.

    Subjects are partitioned by :func:`make_identity_split` into training,
    enrolled (gallery + known probes), and unknown (probe-only) roles.
    Training uses ``train_amp`` with contiguous labels and P×K batches;
    galleries use enrolled ``train_amp``; probes use ``test_amp``. Evaluation
    datasets carry stable subject indices into ``identity_names``.
    """

    name = "ntu_fi_humanid_reid"
    dirname = "NTU-Fi-HumanID"
    sample_shape = (3, 114, 500)
    task_type = "reid"
    checkpoint_monitor = "val/mAP"
    checkpoint_mode = "max"

    def __init__(
        self,
        root,
        split_seed: int = 42,
        identities_per_batch: int = 4,
        samples_per_identity: int = 4,
        eval_batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__(batch_size=eval_batch_size, num_workers=num_workers)
        self.root = Path(root) / self.dirname
        layout = (
            f"{self.dirname}/\n"
            "├── train_amp/<subject id>/*.mat\n"
            "└── test_amp/<subject id>/*.mat"
        )
        self._require(self.root / "train_amp", layout)
        self._require(self.root / "test_amp", layout)
        self.identity_names = sorted(
            d.name for d in (self.root / "train_amp").iterdir() if d.is_dir()
        )
        self.split_seed = split_seed
        self.identities_per_batch = identities_per_batch
        self.samples_per_identity = samples_per_identity
        self.eval_batch_size = eval_batch_size
        self.split_manifest = make_identity_split(
            self.identity_names, seed=split_seed
        )
        self.class_names = list(self.split_manifest.train)
        self._stable = {name: i for i, name in enumerate(self.identity_names)}

    def _eval_set(self, split: str, identities) -> _MatFolderDataset:
        return _MatFolderDataset(
            self.root / split, list(identities), self._stable
        )

    def setup(self, stage: str | None = None) -> None:
        manifest = self.split_manifest
        self.train_set = _MatFolderDataset(
            self.root / "train_amp", self.class_names
        )
        self.val_sets = {
            "gallery": self._eval_set("train_amp", manifest.val_enrolled),
            "known_probes": self._eval_set("test_amp", manifest.val_enrolled),
            "unknown_probes": self._eval_set("test_amp", manifest.val_unknown),
        }
        self.test_sets = {
            "gallery": self._eval_set("train_amp", manifest.test_enrolled),
            "known_probes": self._eval_set("test_amp", manifest.test_enrolled),
            "unknown_probes": self._eval_set("test_amp", manifest.test_unknown),
        }

    def train_dataloader(self) -> DataLoader:
        sampler = IdentityBatchSampler(
            self.train_set.labels,
            self.identities_per_batch,
            self.samples_per_identity,
            seed=self.split_seed,
        )
        return DataLoader(
            self.train_set,
            batch_sampler=sampler,
            num_workers=self.num_workers,
        )

    def validation_loaders_by_role(self) -> dict[str, DataLoader]:
        return {role: self._loader(ds) for role, ds in self.val_sets.items()}

    def test_loaders_by_role(self) -> dict[str, DataLoader]:
        return {role: self._loader(ds) for role, ds in self.test_sets.items()}

    def val_dataloader(self) -> list[DataLoader]:
        # Gallery first, known probes second: ReIDModule computes val mAP
        # from exactly this ordering.
        return [
            self._loader(self.val_sets["gallery"]),
            self._loader(self.val_sets["known_probes"]),
        ]