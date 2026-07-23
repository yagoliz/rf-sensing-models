"""WiMANS people-count classification and regression data."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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
