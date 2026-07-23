import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from rfsensing import data
from rfsensing.data.wimans import (
    _WiMANSDataset,
    _load_records,
    _preprocess_amplitude,
)
from tests.conftest import DATA_ROOT, requires_wimans


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


def test_preprocess_left_pads_and_mean_pools(tmp_path):
    path = tmp_path / "sample.npy"
    values = np.broadcast_to(
        np.arange(1, 4, dtype=np.float32)[:, None, None, None],
        (3, 3, 3, 30),
    ).copy()
    np.save(path, values)
    output = _preprocess_amplitude(
        path,
        raw_time_steps=4,
        time_steps=2,
        pad_side="left",
        pooling="mean",
    )
    assert output.shape == (9, 30, 2)
    assert torch.allclose(output[0, 0], torch.tensor([0.5, 2.5]))


def test_preprocess_supports_full_length_and_max_pooling(tmp_path):
    path = tmp_path / "sample.npy"
    np.save(path, np.ones((3, 3, 3, 30), dtype=np.float32))
    full = _preprocess_amplitude(path, 4, None, "right", "mean")
    pooled = _preprocess_amplitude(path, 4, 2, "right", "max")
    assert full.shape == (9, 30, 4)
    assert pooled.shape == (9, 30, 2)


def test_preprocess_rejects_bad_shape_and_nonfinite_values(tmp_path):
    bad_shape = tmp_path / "bad-shape.npy"
    nonfinite = tmp_path / "nonfinite.npy"
    np.save(bad_shape, np.ones((4, 3, 30), dtype=np.float32))
    values = np.ones((4, 3, 3, 30), dtype=np.float32)
    values[0, 0, 0, 0] = np.nan
    np.save(nonfinite, values)
    with pytest.raises(ValueError, match="expected"):
        _preprocess_amplitude(bad_shape, 4, 2, "left", "mean")
    with pytest.raises(ValueError, match="non-finite"):
        _preprocess_amplitude(nonfinite, 4, 2, "left", "mean")


def test_preprocess_rejects_empty_trace(tmp_path):
    empty = tmp_path / "empty.npy"
    np.save(empty, np.empty((0, 3, 3, 30), dtype=np.float32))
    with pytest.raises(ValueError, match=r"empty\.npy.*positive time length"):
        _preprocess_amplitude(empty, 4, 2, "left", "mean")


@pytest.mark.parametrize(
    "target,dtype",
    [("classification", torch.int64), ("regression", torch.float32)],
)
def test_lazy_dataset_emits_task_specific_targets(tmp_path, target, dtype):
    _make_wimans_tree(
        tmp_path, groups_per_count=1, samples_per_group=1
    )
    records = _load_records(tmp_path)
    dataset = _WiMANSDataset(
        records,
        target=target,
        raw_time_steps=12,
        time_steps=3,
        pad_side="left",
        pooling="mean",
        normalization="none",
    )
    x, y = dataset[0]
    assert x.shape == (9, 30, 3)
    assert x.dtype == torch.float32
    assert y.dtype == dtype


def test_sample_normalization_is_per_link(tmp_path):
    _make_wimans_tree(
        tmp_path, groups_per_count=1, samples_per_group=1
    )
    dataset = _WiMANSDataset(
        _load_records(tmp_path),
        target="classification",
        raw_time_steps=12,
        time_steps=3,
        pad_side="left",
        pooling="mean",
        normalization="sample",
    )
    x, _ = dataset[0]
    assert torch.allclose(
        x.mean(dim=(1, 2)), torch.zeros(9), atol=1e-5
    )
    assert torch.allclose(
        x.std(dim=(1, 2), unbiased=False), torch.ones(9), atol=1e-5
    )


def test_wimans_group_split_is_deterministic_stratified_and_disjoint(tmp_path):
    _make_wimans_tree(tmp_path)
    first = data.build(
        "wimans",
        root=tmp_path,
        target="classification",
        raw_time_steps=12,
        time_steps=3,
        normalization="none",
        split_seed=17,
        batch_size=4,
    )
    second = data.build(
        "wimans",
        root=tmp_path,
        target="classification",
        raw_time_steps=12,
        time_steps=3,
        normalization="none",
        split_seed=17,
        batch_size=4,
    )
    first.setup()
    second.setup()
    assert first.split_group_ids == second.split_group_ids
    train = first.split_group_ids["train"]
    val = first.split_group_ids["val"]
    test = first.split_group_ids["test"]
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    for records in first.split_records.values():
        assert {record.count for record in records} == set(range(6))


def test_wimans_target_metadata_and_batches(tmp_path):
    _make_wimans_tree(tmp_path)
    classification = data.build(
        "wimans",
        root=tmp_path,
        target="classification",
        raw_time_steps=12,
        time_steps=3,
        normalization="none",
        batch_size=4,
    )
    regression = data.build(
        "wimans",
        root=tmp_path,
        target="regression",
        raw_time_steps=12,
        time_steps=3,
        normalization="none",
        batch_size=4,
    )
    classification.setup()
    regression.setup()
    assert classification.split_group_ids == regression.split_group_ids
    assert classification.output_dim == 6
    assert classification.ordered_values == tuple(float(i) for i in range(6))
    assert regression.output_dim == 1
    assert regression.task_type == "regression"
    assert regression.checkpoint_monitor == "val/mae"
    x_class, y_class = next(iter(classification.train_dataloader()))
    x_reg, y_reg = next(iter(regression.train_dataloader()))
    assert x_class.shape[1:] == x_reg.shape[1:] == (9, 30, 3)
    assert torch.equal(
        classification.train_set[0][0], regression.train_set[0][0]
    )
    assert y_class.dtype == torch.int64
    assert y_reg.dtype == torch.float32


def test_wimans_training_normalization_uses_training_records(tmp_path):
    _make_wimans_tree(tmp_path)
    dm = data.build(
        "wimans",
        root=tmp_path,
        raw_time_steps=12,
        time_steps=3,
        normalization="train",
        batch_size=16,
    )
    dm.setup()
    xs = torch.stack([dm.train_set[index][0] for index in range(len(dm.train_set))])
    link_means = xs.mean(dim=(0, 2, 3))
    link_stds = xs.std(dim=(0, 2, 3), unbiased=False)
    assert torch.allclose(link_means, torch.zeros(9), atol=1e-5)
    assert torch.allclose(link_stds, torch.ones(9), atol=1e-5)


def test_wimans_rejects_invalid_configuration(tmp_path):
    _make_wimans_tree(tmp_path)
    with pytest.raises(ValueError, match="target"):
        data.build("wimans", root=tmp_path, target="ordinal")
    with pytest.raises(ValueError, match="split_ratios"):
        data.build("wimans", root=tmp_path, split_ratios=(0.8, 0.2, 0.2))
    with pytest.raises(ValueError, match="time_steps"):
        data.build("wimans", root=tmp_path, raw_time_steps=12, time_steps=13)


def test_group_split_rejects_too_few_groups_per_count(tmp_path):
    _make_wimans_tree(tmp_path, groups_per_count=2, samples_per_group=1)
    dm = data.build(
        "wimans",
        root=tmp_path,
        raw_time_steps=12,
        time_steps=3,
        normalization="none",
    )
    with pytest.raises(ValueError, match="at least 3 groups per count"):
        dm.setup()


def test_random_split_is_available_for_paper_comparison(tmp_path):
    _make_wimans_tree(tmp_path)
    dm = data.build(
        "wimans",
        root=tmp_path,
        raw_time_steps=12,
        time_steps=3,
        normalization="none",
        split_strategy="random",
    )
    dm.setup()
    assert sum(len(records) for records in dm.split_records.values()) == 96
    for records in dm.split_records.values():
        assert {record.count for record in records} == set(range(6))


@pytest.mark.data
@requires_wimans
def test_wimans_real_data_contract():
    dm = data.build(
        "wimans",
        root=DATA_ROOT,
        target="classification",
        time_steps=30,
        normalization="none",
        batch_size=2,
    )
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (2, 9, 30, 30)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert dm.output_dim == 6
