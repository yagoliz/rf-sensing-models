"""In-house USRP + ESP32 (+ Nexmon) position captures (``csi-capturing/scripts/dual_capture.py``).

Each kept trial is one directory with an ``aligned.npz`` holding both
sniffers' CSI on a common time grid; ``manifest.jsonl`` lists the kept trials
and their condition label (``empty``, ``pos1``, ...). Samples are amplitude
windows of shape ``(1, subcarriers, window_steps)`` (link, subcarrier, time),
the WiMANS layout with a single antenna link. Splits are by trial: windows of
one trial are strongly correlated, so a trial never spans two splits.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import TensorDataset

from rfsensing.data import register
from rfsensing.data.base import CSIDataModule
from rfsensing.data.wimans import _allocate_group_counts

DEVICES = ("usrp", "esp", "nexmon")  # every sniffer a trial may hold
_LAYOUT = """\
<root>/<subdir>/manifest.jsonl
<root>/<subdir>/<stamp>_<label>_<nn>/aligned.npz
(copy the --output-dir of dual_capture.py, default csi-capturing/data/dual)"""


@dataclass(frozen=True)
class Trial:
    """One kept trial: both sniffers' CSI on the shared grid."""

    name: str
    label: str
    rate: float
    t: np.ndarray  # [T] wall time, seconds since epoch
    csi: dict[str, np.ndarray]  # device -> complex64 [T, S]; only the sniffers recorded
    valid: dict[str, np.ndarray]  # device -> bool [T]
    subcarriers: np.ndarray  # [S]
    manifest: dict

    def amplitude(self, device: str) -> np.ndarray:
        """float32 [T, S] amplitude of one sniffer."""
        return np.abs(self.csi[device]).astype(np.float32)


def load_trials(root: str | Path, subdir: str = "dual") -> list[Trial]:
    """Every trial listed in the manifest, in recording order."""
    base = Path(root) / subdir
    manifest = base / "manifest.jsonl"
    CSIDataModule._require(manifest, _LAYOUT)
    trials = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = base / row["dir"] / "aligned.npz"
        CSIDataModule._require(path, _LAYOUT)
        with np.load(path) as f:
            present = [d for d in DEVICES if d in f.files]
            trials.append(
                Trial(
                    name=row["dir"],
                    label=row["label"],
                    rate=float(f["rate"]),
                    t=f["t"],
                    # stored as [T, tx, rx, S]; both sniffers have one link
                    csi={d: f[d].reshape(len(f["t"]), -1) for d in present},
                    valid={d: f[f"{d}_valid"] for d in present},
                    subcarriers=f["subcarriers"],
                    manifest=row,
                )
            )
    if not trials:
        raise ValueError(f"{manifest} lists no trials")
    return trials


def available_devices(trials: Sequence[Trial]) -> list[str]:
    """Sniffers recorded in every trial, in ``DEVICES`` order."""
    return [d for d in DEVICES if all(d in t.csi for t in trials)]


def window_starts(
    valid: np.ndarray, window_steps: int, stride: int, min_valid: float
) -> np.ndarray:
    """Start indices of windows whose grid points are fresh often enough."""
    starts = np.arange(0, len(valid) - window_steps + 1, stride)
    fresh = np.array([valid[s : s + window_steps].mean() for s in starts])
    return starts[fresh >= min_valid] if len(starts) else starts


def _windows(
    trials: Sequence[Trial],
    device: str,
    class_index: dict[str, int],
    window_steps: int,
    stride: int,
    min_valid: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for trial in trials:
        amp = trial.amplitude(device)
        for s in window_starts(trial.valid[device], window_steps, stride, min_valid):
            xs.append(amp[s : s + window_steps].T[None])  # [1, S, W]
            ys.append(class_index[trial.label])
    if not xs:
        raise ValueError(
            f"no {device} windows of {window_steps} steps with >= {min_valid:.0%} "
            "fresh grid points; lower min_valid or window_steps"
        )
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys, dtype=torch.long)


def _trial_split(
    trials: Sequence[Trial],
    labels: Sequence[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Trial]]:
    """Per label, shuffle its trials and deal them into train/val/test."""
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for label in labels:
        mine = [t for t in trials if t.label == label]
        order = rng.permutation(len(mine))
        n_train, n_val, _ = _allocate_group_counts(len(mine), ratios)
        for rank, i in enumerate(order):
            key = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
            split[key].append(mine[i])
    return split


@register("dual_capture")
class DualCaptureDataModule(CSIDataModule):
    """Position classification from one sniffer of the dual captures."""

    name = "dual_capture"

    def __init__(
        self,
        root,
        *,
        device="usrp",
        subdir="dual",
        labels=None,
        window_steps=300,
        train_stride=None,
        min_valid=0.9,
        normalization="train",
        split_ratios=(0.6, 0.2, 0.2),
        split_seed=42,
        batch_size=32,
        num_workers=0,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        if normalization not in {"train", "sample", "none"}:
            raise ValueError("normalization must be train, sample, or none")
        if not 0.0 <= min_valid <= 1.0:
            raise ValueError("min_valid must be in [0, 1]")
        split_ratios = tuple(float(v) for v in split_ratios)
        if (
            len(split_ratios) != 3
            or any(v <= 0 for v in split_ratios)
            or not np.isclose(sum(split_ratios), 1.0)
        ):
            raise ValueError("split_ratios must contain 3 positive values summing to 1")
        self.device = device
        self.window_steps = window_steps
        # Overlapping training windows multiply the samples; evaluation windows
        # do not overlap, so each test step is scored once.
        self.train_stride = train_stride or window_steps // 2
        self.min_valid = min_valid
        self.normalization = normalization
        self.split_ratios = split_ratios
        self.split_seed = split_seed
        self.trials = load_trials(root, subdir)
        found = {t.label for t in self.trials}
        if labels is None:
            labels = sorted(found, key=lambda lab: (lab != "empty", lab))
        missing = set(labels) - found
        if missing:
            raise ValueError(f"labels {sorted(missing)} have no trials")
        self.trials = [t for t in self.trials if t.label in labels]
        lacking = [t.name for t in self.trials if device not in t.csi]
        if lacking:
            raise ValueError(f"{len(lacking)} trials have no {device} data, e.g. {lacking[0]}")
        self.class_names = list(labels)
        n_sub = self.trials[0].csi[device].shape[1]
        self.sample_shape = (1, n_sub, window_steps)
        self._is_setup = False

    def setup(self, stage=None):
        if self._is_setup:
            return
        self.split_trials = _trial_split(
            self.trials, self.class_names, self.split_ratios, self.split_seed
        )
        self._build_sets({label: i for i, label in enumerate(self.class_names)})
        self._is_setup = True

    def _build_sets(self, index: dict[str, int]) -> None:
        """Window ``self.split_trials`` into ``{train,val,test}_set``."""
        tensors = {
            split: _windows(
                trials,
                self.device,
                index,
                self.window_steps,
                self.train_stride if split == "train" else self.window_steps,
                self.min_valid,
            )
            for split, trials in self.split_trials.items()
        }
        if self.normalization == "train":
            x_train = tensors["train"][0]
            mean = x_train.mean(dim=(0, 3), keepdim=True)[0]  # per subcarrier
            std = x_train.std(dim=(0, 3), keepdim=True)[0].clamp_min(1e-6)
        for split, (x, y) in tensors.items():
            if self.normalization == "train":
                x = (x - mean) / std
            elif self.normalization == "sample":
                # per window: removes gain/AGC level shifts between trials
                mu = x.mean(dim=(1, 2, 3), keepdim=True)
                sd = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                x = (x - mu) / sd
            setattr(self, f"{split}_set", TensorDataset(x, y))

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)


@register("dual_capture_anomaly")
class DualCaptureAnomalyDataModule(DualCaptureDataModule):
    """Novelty detection: learn one condition, flag every other one.

    Only ``normal_label`` trials (default ``empty``) are split train/val/test;
    every other trial is held out entirely and joins the test split, so the
    model never sees an anomaly. Targets are 0 (normal) / 1 (anomaly);
    ``test_labels`` and ``test_trial_names`` give each test window's condition
    and trial for per-position scores.
    """

    name = "dual_capture_anomaly"

    def __init__(self, root, *, normal_label="empty", **kwargs):
        super().__init__(root, **kwargs)
        if normal_label not in self.class_names:
            raise ValueError(f"normal_label {normal_label!r} has no trials")
        self.normal_label = normal_label
        self.anomaly_labels = [lab for lab in self.class_names if lab != normal_label]
        if not self.anomaly_labels:
            raise ValueError("anomaly detection needs trials of at least one other label")
        self.class_names = ["normal", "anomaly"]

    def setup(self, stage=None):
        if self._is_setup:
            return
        split = _trial_split(self.trials, [self.normal_label], self.split_ratios, self.split_seed)
        split["test"] += [t for t in self.trials if t.label != self.normal_label]
        self.split_trials = split
        self._build_sets({self.normal_label: 0} | {lab: 1 for lab in self.anomaly_labels})
        # _windows walks trials and their window starts in order
        counts = [
            len(window_starts(t.valid[self.device], self.window_steps, self.window_steps, self.min_valid))
            for t in split["test"]
        ]
        self.test_labels = np.repeat([t.label for t in split["test"]], counts)
        self.test_trial_names = np.repeat([t.name for t in split["test"]], counts)
        self._is_setup = True
