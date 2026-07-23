import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from rfsensing.data.wimans import _load_records


def _make_wimans_tree(
    root: Path,
    *,
    official_layout: bool = True,
    groups_per_count: int = 8,
    samples_per_group: int = 2,
) -> Path:
    dataset_root = root / "WiMANS"
    amp_dir = (
        dataset_root / "wifi_csi" / "amp"
        if official_layout
        else dataset_root / "amp"
    )
    amp_dir.mkdir(parents=True)
    fields = ["#", "label", "environment", "wifi_band", "number_of_users"]
    rows = []
    group_id = 1
    for count in range(6):
        for local_group in range(groups_per_count):
            environment = ("classroom", "meeting_room", "empty_room")[
                local_group % 3
            ]
            wifi_band = ("2.4", "5")[local_group % 2]
            for sample_id in range(1, samples_per_group + 1):
                label = f"act_{group_id}_{sample_id}"
                rows.append(
                    {
                        "#": str(len(rows) + 1),
                        "label": label,
                        "environment": environment,
                        "wifi_band": wifi_band,
                        "number_of_users": str(count),
                    }
                )
                length = 8 + sample_id
                values = np.arange(
                    length * 3 * 3 * 30, dtype=np.float32
                ).reshape(length, 3, 3, 30)
                np.save(amp_dir / f"{label}.npy", values + group_id)
            group_id += 1
    with (dataset_root / "annotation.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return dataset_root


def test_load_records_accepts_official_and_flat_layouts(tmp_path):
    official_root = tmp_path / "official"
    flat_root = tmp_path / "flat"
    _make_wimans_tree(official_root, official_layout=True)
    _make_wimans_tree(flat_root, official_layout=False)
    official = _load_records(official_root)
    flat = _load_records(flat_root)
    assert len(official) == len(flat) == 96
    assert "wifi_csi/amp" in official[0].path.as_posix()
    assert flat[0].path.parent.name == "amp"


def test_official_layout_takes_precedence(tmp_path):
    _make_wimans_tree(tmp_path, official_layout=True)
    _make_wimans_tree(tmp_path, official_layout=False)
    records = _load_records(tmp_path)
    assert all("wifi_csi/amp" in record.path.as_posix() for record in records)


def test_load_records_filters_annotations(tmp_path):
    _make_wimans_tree(tmp_path)
    records = _load_records(
        tmp_path, environments=("classroom",), wifi_bands=("5",)
    )
    assert records
    assert {record.environment for record in records} == {"classroom"}
    assert {record.wifi_band for record in records} == {"5"}


def test_load_records_reports_incomplete_download(tmp_path):
    dataset_root = _make_wimans_tree(tmp_path)
    missing = dataset_root / "wifi_csi" / "amp" / "act_1_1.npy"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="1 missing"):
        _load_records(tmp_path)
    records = _load_records(tmp_path, allow_partial=True)
    assert len(records) == 95
