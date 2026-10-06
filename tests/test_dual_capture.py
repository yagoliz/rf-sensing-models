import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rfsensing import data
from rfsensing.data.dual_capture import available_devices, load_trials, window_starts

LABELS = ("pos1", "empty", "pos2")


def _make_dual_tree(
    root: Path,
    *,
    trials_per_label: int = 5,
    steps: int = 1000,
    n_sub: int = 30,
    devices: tuple[str, ...] = ("usrp", "esp"),
) -> Path:
    base = root / "dual"
    base.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(trials_per_label):
        for li, label in enumerate(LABELS):
            name = f"20261005_1200{k}{li}_{label}_{k + 1:02d}"
            (base / name).mkdir()
            level = 1.0 + li  # each label gets its own amplitude level
            csi = {
                d: (level + 0.1 * rng.standard_normal((steps, 1, 1, n_sub))).astype(
                    np.complex64
                )
                for d in devices
            }
            esp_valid = np.ones(steps, bool)
            esp_valid[:150] = False  # a stale start: the first window is dropped
            valid = {d: np.ones(steps, bool) for d in devices} | {"esp": esp_valid}
            np.savez_compressed(
                base / name / "aligned.npz",
                t=1e9 + np.arange(steps) / 100.0,
                **csi,
                **{f"{d}_valid": valid[d] for d in devices},
                subcarriers=np.arange(n_sub),
                rate=100.0,
                label=label,
            )
            rows.append({"label": label, "dir": name, "windows": steps // 300})
    (base / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_load_trials_reads_manifest_order(tmp_path):
    trials = load_trials(_make_dual_tree(tmp_path))
    assert len(trials) == 15
    assert trials[0].label == "pos1"
    assert trials[0].csi["usrp"].shape == (1000, 30)
    assert trials[0].amplitude("esp").dtype == np.float32


def test_missing_manifest_names_the_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.jsonl"):
        load_trials(tmp_path)


def test_window_starts_drop_stale_windows():
    valid = np.ones(1000, bool)
    valid[:150] = False
    assert window_starts(valid, 300, 300, 0.9).tolist() == [300, 600]
    assert window_starts(valid, 300, 150, 0.0).tolist() == [0, 150, 300, 450, 600]


def test_datamodule_splits_by_trial(tmp_path):
    dm = data.build("dual_capture", root=_make_dual_tree(tmp_path), device="usrp")
    dm.setup()
    assert dm.class_names == ["empty", "pos1", "pos2"]
    assert dm.sample_shape == (1, 30, 300)
    names = {s: {t.name for t in ts} for s, ts in dm.split_trials.items()}
    assert names["train"].isdisjoint(names["val"])
    assert names["train"].isdisjoint(names["test"])
    assert names["val"].isdisjoint(names["test"])
    for split, trials in dm.split_trials.items():
        per_label = [sum(t.label == lab for t in trials) for lab in dm.class_names]
        assert per_label == [{"train": 3, "val": 1, "test": 1}[split]] * 3
    x, y = dm.train_set[0]
    assert x.shape == (1, 30, 300) and x.dtype == torch.float32
    assert y.dtype == torch.long
    # train windows overlap by half, evaluation windows do not
    assert len(dm.train_set) == 9 * 5
    assert len(dm.test_set) == 3 * 3


def test_esp_drops_stale_windows_and_normalizes(tmp_path):
    root = _make_dual_tree(tmp_path)
    dm = data.build("dual_capture", root=root, device="esp", normalization="sample")
    dm.setup()
    assert len(dm.test_set) == 3 * 2  # window at step 0 is < 90 % fresh
    x = dm.test_set.tensors[0]
    assert torch.allclose(x.mean(dim=(1, 2, 3)), torch.zeros(len(x)), atol=1e-5)

    dm = data.build("dual_capture", root=root, device="esp", normalization="train")
    dm.setup()
    x = dm.train_set.tensors[0]
    assert torch.allclose(x.mean(dim=(0, 3)), torch.zeros(1, 30), atol=1e-4)


def test_label_subset_and_validation(tmp_path):
    root = _make_dual_tree(tmp_path)
    dm = data.build("dual_capture", root=root, labels=["pos1", "pos2"])
    assert dm.num_classes == 2 and len(dm.trials) == 10
    with pytest.raises(ValueError, match="no trials"):
        data.build("dual_capture", root=root, labels=["pos3"])
    with pytest.raises(ValueError, match="device"):
        data.build("dual_capture", root=root, device="ax210")


def test_nexmon_is_optional_and_detected(tmp_path):
    two = load_trials(_make_dual_tree(tmp_path / "two"))
    assert available_devices(two) == ["usrp", "esp"]
    with pytest.raises(ValueError, match="no nexmon data"):
        data.build("dual_capture", root=tmp_path / "two", device="nexmon")

    root = _make_dual_tree(tmp_path / "three", devices=("usrp", "esp", "nexmon"))
    assert available_devices(load_trials(root)) == ["usrp", "esp", "nexmon"]
    dm = data.build("dual_capture", root=root, device="nexmon")
    dm.setup()
    assert dm.sample_shape == (1, 30, 300) and len(dm.test_set) == 9


def test_anomaly_datamodule_trains_on_normal_only(tmp_path):
    root = _make_dual_tree(tmp_path)
    dm = data.build("dual_capture_anomaly", root=root, device="esp", labels=["empty", "pos1"])
    dm.setup()
    assert dm.class_names == ["normal", "anomaly"] and dm.anomaly_labels == ["pos1"]
    for split in ("train", "val"):
        assert {t.label for t in dm.split_trials[split]} == {"empty"}
        assert dm.__dict__[f"{split}_set"].tensors[1].eq(0).all()
    names = {s: {t.name for t in ts} for s, ts in dm.split_trials.items()}
    assert names["train"].isdisjoint(names["test"]) and names["val"].isdisjoint(names["test"])
    # 1 held-out empty trial + all 5 pos1 trials, 2 fresh esp windows each
    y = dm.test_set.tensors[1]
    assert len(y) == len(dm.test_labels) == len(dm.test_trial_names) == 6 * 2
    assert np.array_equal(y.numpy(), (dm.test_labels != "empty").astype(int))
    assert set(dm.test_trial_names[dm.test_labels == "empty"]) <= names["test"]


def test_anomaly_datamodule_validation(tmp_path):
    root = _make_dual_tree(tmp_path)
    with pytest.raises(ValueError, match="normal_label"):
        data.build("dual_capture_anomaly", root=root, labels=["pos1", "pos2"])
    with pytest.raises(ValueError, match="at least one other"):
        data.build("dual_capture_anomaly", root=root, labels=["empty"])
