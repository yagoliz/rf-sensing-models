"""WiMANS people-count classification and regression data."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
