"""WiMANS people-count classification and regression data."""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from rfsensing.data import register
from rfsensing.data.base import CSIDataModule

_DOWNLOAD_URL = "https://www.kaggle.com/datasets/shuokanghuang/wimans"
_LABEL_PATTERN = re.compile(r"act_(\d+)_(\d+)")
_REQUIRED_COLUMNS = {
    "label",
    "environment",
    "wifi_band",
    "number_of_users",
}


@dataclass(frozen=True)
class WiMANSRecord:
    label: str
    group_id: int
    sample_id: int
    environment: str
    wifi_band: str
    count: int
    path: Path


def _resolve_amp_dir(dataset_root: Path) -> Path:
    official = dataset_root / "wifi_csi" / "amp"
    flat = dataset_root / "amp"
    if official.is_dir():
        return official
    if flat.is_dir():
        return flat
    raise FileNotFoundError(
        f"No WiMANS amplitude directory found under {dataset_root}. "
        "Expected wifi_csi/amp/ or amp/. "
        f"Download the complete dataset from {_DOWNLOAD_URL}"
    )


def _load_records(
    root: str | Path,
    environments: Sequence[str] | None = None,
    wifi_bands: Sequence[str] | None = None,
    allow_partial: bool = False,
) -> list[WiMANSRecord]:
    dataset_root = Path(root) / "WiMANS"
    annotation_path = dataset_root / "annotation.csv"
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"{annotation_path} not found. Download WiMANS from {_DOWNLOAD_URL}"
        )
    amp_dir = _resolve_amp_dir(dataset_root)
    environment_filter = set(environments) if environments is not None else None
    band_filter = set(wifi_bands) if wifi_bands is not None else None
    records = []
    with annotation_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing_columns = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{annotation_path} is missing columns {sorted(missing_columns)}"
            )
        for row in reader:
            if (
                environment_filter is not None
                and row["environment"] not in environment_filter
            ):
                continue
            if band_filter is not None and row["wifi_band"] not in band_filter:
                continue
            match = _LABEL_PATTERN.fullmatch(row["label"])
            if match is None:
                raise ValueError(f"Invalid WiMANS label {row['label']!r}")
            count = int(row["number_of_users"])
            if count not in range(6):
                raise ValueError(f"Invalid WiMANS user count {count}")
            records.append(
                WiMANSRecord(
                    label=row["label"],
                    group_id=int(match.group(1)),
                    sample_id=int(match.group(2)),
                    environment=row["environment"],
                    wifi_band=row["wifi_band"],
                    count=count,
                    path=amp_dir / f"{row['label']}.npy",
                )
            )
    missing = [record.path for record in records if not record.path.is_file()]
    if missing and not allow_partial:
        examples = "\n".join(f"- {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"WiMANS selection has {len(records)} annotations, "
            f"{len(records) - len(missing)} present files, and "
            f"{len(missing)} missing amplitude files. Examples:\n{examples}\n"
            f"Download the complete dataset from {_DOWNLOAD_URL}"
        )
    if allow_partial:
        records = [record for record in records if record.path.is_file()]
    if not records:
        raise ValueError("WiMANS filters selected no samples")
    return records


def _preprocess_amplitude(
    path: Path,
    raw_time_steps: int,
    time_steps: int | None,
    pad_side: str,
    pooling: str,
) -> torch.Tensor:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 4 or values.shape[1:] != (3, 3, 30):
        raise ValueError(
            f"{path} has shape {values.shape}; expected (time, 3, 3, 30)"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite amplitude values")
    values = values.astype(np.float32, copy=False)
    if values.shape[0] > raw_time_steps:
        values = (
            values[-raw_time_steps:]
            if pad_side == "left"
            else values[:raw_time_steps]
        )
    padding = raw_time_steps - values.shape[0]
    widths = (
        ((padding, 0), (0, 0), (0, 0), (0, 0))
        if pad_side == "left"
        else ((0, padding), (0, 0), (0, 0), (0, 0))
    )
    values = np.pad(values, widths)
    tensor = torch.from_numpy(
        values.transpose(1, 2, 3, 0).reshape(9, 30, raw_time_steps)
    )
    if time_steps is None:
        return tensor
    if pooling == "mean":
        return F.adaptive_avg_pool1d(tensor, time_steps)
    if pooling == "max":
        return F.adaptive_max_pool1d(tensor, time_steps)
    raise ValueError(f"Unsupported pooling {pooling!r}")


def _fit_link_stats(
    records: Sequence[WiMANSRecord],
    raw_time_steps: int,
    time_steps: int | None,
    pad_side: str,
    pooling: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(9, dtype=torch.float64)
    total_sq = torch.zeros(9, dtype=torch.float64)
    values_per_link = 0
    for record in records:
        tensor = _preprocess_amplitude(
            record.path, raw_time_steps, time_steps, pad_side, pooling
        ).double()
        total += tensor.sum(dim=(1, 2))
        total_sq += tensor.square().sum(dim=(1, 2))
        values_per_link += tensor.shape[1] * tensor.shape[2]
    mean = total / values_per_link
    variance = (total_sq / values_per_link - mean.square()).clamp_min(0)
    std = variance.sqrt().clamp_min(1e-6)
    return mean.float()[:, None, None], std.float()[:, None, None]


class _WiMANSDataset(Dataset):
    def __init__(
        self,
        records: Sequence[WiMANSRecord],
        *,
        target: str,
        raw_time_steps: int,
        time_steps: int | None,
        pad_side: str,
        pooling: str,
        normalization: str,
        link_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        self.records = list(records)
        self.target = target
        self.raw_time_steps = raw_time_steps
        self.time_steps = time_steps
        self.pad_side = pad_side
        self.pooling = pooling
        self.normalization = normalization
        self.link_stats = link_stats

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        x = _preprocess_amplitude(
            record.path,
            self.raw_time_steps,
            self.time_steps,
            self.pad_side,
            self.pooling,
        )
        if self.normalization == "sample":
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
            x = (x - mean) / std
        elif self.normalization == "train":
            if self.link_stats is None:
                raise RuntimeError("training normalization statistics are missing")
            mean, std = self.link_stats
            x = (x - mean) / std
        y = torch.tensor(
            record.count,
            dtype=torch.long if self.target == "classification" else torch.float32,
        )
        return x, y


def _allocate_group_counts(
    size: int, ratios: tuple[float, float, float]
) -> tuple[int, int, int]:
    if size < 3:
        raise ValueError(
            f"group splitting requires at least 3 groups per count, got {size}"
        )
    raw = np.asarray(ratios) * size
    allocated = np.floor(raw).astype(int)
    allocated[allocated == 0] = 1
    while allocated.sum() > size:
        index = int(np.argmax(allocated))
        if allocated[index] == 1:
            raise ValueError("cannot allocate every count to all three splits")
        allocated[index] -= 1
    while allocated.sum() < size:
        fractions = raw - np.floor(raw)
        index = int(np.argmax(fractions - allocated * 1e-12))
        allocated[index] += 1
    return tuple(int(value) for value in allocated)


def _group_split(
    records: Sequence[WiMANSRecord],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[WiMANSRecord]]:
    count_by_group = {}
    groups_by_count = defaultdict(list)
    for record in records:
        previous = count_by_group.setdefault(record.group_id, record.count)
        if previous != record.count:
            raise ValueError(
                f"WiMANS group {record.group_id} has inconsistent counts"
            )
    for group_id, count in count_by_group.items():
        groups_by_count[count].append(group_id)
    split_groups = {"train": set(), "val": set(), "test": set()}
    rng = np.random.default_rng(seed)
    for count in range(6):
        groups = np.asarray(sorted(groups_by_count[count]))
        rng.shuffle(groups)
        n_train, n_val, n_test = _allocate_group_counts(len(groups), ratios)
        split_groups["train"].update(groups[:n_train].tolist())
        split_groups["val"].update(groups[n_train : n_train + n_val].tolist())
        split_groups["test"].update(groups[-n_test:].tolist())
    return {
        split: [
            record for record in records if record.group_id in group_ids
        ]
        for split, group_ids in split_groups.items()
    }


def _random_split(
    records: Sequence[WiMANSRecord],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[WiMANSRecord]]:
    indices = np.arange(len(records))
    counts = np.asarray([record.count for record in records])
    train_indices, remainder = train_test_split(
        indices,
        train_size=ratios[0],
        random_state=seed,
        stratify=counts,
    )
    relative_val = ratios[1] / (ratios[1] + ratios[2])
    val_indices, test_indices = train_test_split(
        remainder,
        train_size=relative_val,
        random_state=seed,
        stratify=counts[remainder],
    )
    return {
        "train": [records[index] for index in train_indices],
        "val": [records[index] for index in val_indices],
        "test": [records[index] for index in test_indices],
    }


@register("wimans")
class WiMANSDataModule(CSIDataModule):
    """WiMANS group-size classification or scalar count regression."""

    name = "wimans"
    class_names = [str(value) for value in range(6)]
    target_range = (0.0, 5.0)

    def __init__(
        self,
        root,
        *,
        target="classification",
        raw_time_steps=3000,
        time_steps=300,
        pad_side="left",
        pooling="mean",
        normalization="train",
        split_strategy="group",
        split_ratios=(0.70, 0.15, 0.15),
        split_seed=42,
        environments=None,
        wifi_bands=None,
        allow_partial=False,
        batch_size=32,
        num_workers=0,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        if target not in {"classification", "regression"}:
            raise ValueError(
                "target must be classification or regression, "
                f"got {target!r}"
            )
        if raw_time_steps <= 0:
            raise ValueError("raw_time_steps must be positive")
        if time_steps is not None and not 1 <= time_steps <= raw_time_steps:
            raise ValueError("time_steps must be between 1 and raw_time_steps")
        if pad_side not in {"left", "right"}:
            raise ValueError("pad_side must be left or right")
        if pooling not in {"mean", "max"}:
            raise ValueError("pooling must be mean or max")
        if normalization not in {"train", "sample", "none"}:
            raise ValueError("normalization must be train, sample, or none")
        if split_strategy not in {"group", "random"}:
            raise ValueError("split_strategy must be group or random")
        split_ratios = tuple(float(value) for value in split_ratios)
        if (
            len(split_ratios) != 3
            or any(value <= 0 for value in split_ratios)
            or not np.isclose(sum(split_ratios), 1.0)
        ):
            raise ValueError(
                "split_ratios must contain 3 positive values summing to 1"
            )
        self.target = target
        self.task_type = target
        self.ordered_values = (
            tuple(float(value) for value in range(6))
            if target == "classification"
            else None
        )
        self.checkpoint_monitor = (
            "val/acc" if target == "classification" else "val/mae"
        )
        self.checkpoint_mode = "max" if target == "classification" else "min"
        self.raw_time_steps = raw_time_steps
        self.time_steps = time_steps
        self.sample_shape = (9, 30, time_steps or raw_time_steps)
        self.pad_side = pad_side
        self.pooling = pooling
        self.normalization = normalization
        self.split_strategy = split_strategy
        self.split_ratios = split_ratios
        self.split_seed = split_seed
        self.records = _load_records(
            root,
            environments=environments,
            wifi_bands=wifi_bands,
            allow_partial=allow_partial,
        )
        self._is_setup = False

    @property
    def output_dim(self):
        return 6 if self.target == "classification" else 1

    def setup(self, stage=None):
        if self._is_setup:
            return
        split_records = (
            _group_split(self.records, self.split_ratios, self.split_seed)
            if self.split_strategy == "group"
            else _random_split(self.records, self.split_ratios, self.split_seed)
        )
        link_stats = None
        if self.normalization == "train":
            link_stats = _fit_link_stats(
                split_records["train"],
                self.raw_time_steps,
                self.time_steps,
                self.pad_side,
                self.pooling,
            )
        dataset_kwargs = {
            "target": self.target,
            "raw_time_steps": self.raw_time_steps,
            "time_steps": self.time_steps,
            "pad_side": self.pad_side,
            "pooling": self.pooling,
            "normalization": self.normalization,
            "link_stats": link_stats,
        }
        self.split_records = split_records
        self.split_group_ids = {
            split: {record.group_id for record in records}
            for split, records in split_records.items()
        }
        self.train_set = _WiMANSDataset(split_records["train"], **dataset_kwargs)
        self.val_set = _WiMANSDataset(split_records["val"], **dataset_kwargs)
        self.test_set = _WiMANSDataset(split_records["test"], **dataset_kwargs)
        self._is_setup = True

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)
