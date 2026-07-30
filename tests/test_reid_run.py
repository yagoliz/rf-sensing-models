import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from rfsensing import models
from rfsensing.data.base import CSIDataModule
from rfsensing.data.reid import IdentityBatchSampler, make_identity_split
from rfsensing.eval.reid import calibrate_thresholds, score_gallery_probe
from rfsensing.train.reid_run import (
    ReIDResult,
    _embed_loader,
    _predictions_rows,
    _save_json,
    run_reid,
    run_reid_repeats,
)


class _TinyReIDDataModule(CSIDataModule):
    name = "tiny_reid"
    sample_shape = (3, 8, 8)
    task_type = "reid"
    checkpoint_monitor = "val/mAP"
    checkpoint_mode = "max"

    def __init__(self, split_seed: int = 42, samples: int = 4):
        super().__init__(batch_size=8)
        self.split_seed = split_seed
        self.samples = samples
        self.identity_names = [f"s{i:02d}" for i in range(1, 15)]
        self.split_manifest = make_identity_split(
            self.identity_names, seed=split_seed
        )
        self.class_names = list(self.split_manifest.train)
        self._stable = {n: i for i, n in enumerate(self.identity_names)}

    @property
    def output_dim(self) -> int:
        return len(self.class_names)

    def _dataset(self, names, label_of):
        generator = torch.Generator().manual_seed(
            self.split_seed + sum(self._stable[n] for n in names)
        )
        xs, ys = [], []
        for name in names:
            center = torch.randn(
                self.sample_shape,
                generator=torch.Generator().manual_seed(self._stable[name]),
            )
            for _ in range(self.samples):
                xs.append(
                    center
                    + 0.05 * torch.randn(self.sample_shape, generator=generator)
                )
                ys.append(label_of(name))
        return TensorDataset(torch.stack(xs), torch.tensor(ys))

    def setup(self, stage=None):
        manifest = self.split_manifest
        contiguous = {n: i for i, n in enumerate(self.class_names)}
        self.train_set = self._dataset(
            manifest.train, lambda n: contiguous[n]
        )
        stable = lambda n: self._stable[n]
        self.val_sets = {
            "gallery": self._dataset(manifest.val_enrolled, stable),
            "known_probes": self._dataset(manifest.val_enrolled, stable),
            "unknown_probes": self._dataset(manifest.val_unknown, stable),
        }
        self.test_sets = {
            "gallery": self._dataset(manifest.test_enrolled, stable),
            "known_probes": self._dataset(manifest.test_enrolled, stable),
            "unknown_probes": self._dataset(manifest.test_unknown, stable),
        }

    def train_dataloader(self):
        labels = self.train_set.tensors[1].tolist()
        return DataLoader(
            self.train_set,
            batch_sampler=IdentityBatchSampler(labels, 3, 2, seed=self.split_seed),
        )

    def validation_loaders_by_role(self):
        return {role: self._loader(ds) for role, ds in self.val_sets.items()}

    def test_loaders_by_role(self):
        return {role: self._loader(ds) for role, ds in self.test_sets.items()}

    def val_dataloader(self):
        return [
            self._loader(self.val_sets["gallery"]),
            self._loader(self.val_sets["known_probes"]),
        ]


def _tiny_net(dm):
    return models.build(
        "mlp",
        in_shape=dm.sample_shape,
        num_classes=dm.output_dim,
        hidden_dims=(16,),
    )


# --- artifact helpers ---


def test_save_json_rejects_non_finite(tmp_path):
    path = tmp_path / "out.json"
    _save_json(path, {"ok": 1.0})
    assert json.loads(path.read_text()) == {"ok": 1.0}
    with pytest.raises(ValueError):
        _save_json(path, {"bad": float("nan")})


def test_predictions_rows_contract():
    gallery = torch.eye(2)
    scores = score_gallery_probe(
        gallery, torch.tensor([3, 7]), torch.tensor([[1.0, 0.0], [0.7, 0.7]])
    )
    names = [f"s{i}" for i in range(10)]
    rows = _predictions_rows(
        names,
        scores,
        probe_labels=torch.tensor([3, 9]),
        known_mask=torch.tensor([True, False]),
        operating_points={"eer_threshold": 0.9, "far05_threshold": 1.5},
    )
    assert len(rows) == 2
    first, second = rows
    assert first["probe_identity"] == "s3"
    assert first["known"] is True
    assert first["top_identity"] == "s3"
    assert first["top_score"] == pytest.approx(1.0)
    assert first["rank"] == 1
    assert first["pred_eer_threshold"] == "s3"
    assert first["pred_far05_threshold"] == "unknown"
    assert second["probe_identity"] == "s9"
    assert second["known"] is False
    assert second["rank"] == -1
    assert second["pred_eer_threshold"] == "unknown"


def test_embed_loader_restores_training_mode():
    dm = _TinyReIDDataModule()
    dm.setup()
    net = _tiny_net(dm)
    net.train()
    loader = dm.test_loaders_by_role()["gallery"]
    embeddings, labels = _embed_loader(net, loader)
    assert net.training
    assert embeddings.shape[0] == labels.shape[0] == len(loader.dataset)
    assert torch.isfinite(embeddings).all()


def test_embed_loader_rejects_non_finite():
    dm = _TinyReIDDataModule()
    dm.setup()
    net = _tiny_net(dm)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(float("nan"))
    loader = dm.test_loaders_by_role()["gallery"]
    with pytest.raises(ValueError, match="finite"):
        _embed_loader(net, loader)


# --- run_reid ---


@pytest.fixture(scope="module")
def reid_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("reid_run")
    dm = _TinyReIDDataModule(split_seed=42)
    net = _tiny_net(dm)
    result = run_reid(
        net,
        dm,
        max_epochs=1,
        name="smoke",
        seed=42,
        accelerator="cpu",
        runs_dir=tmp_path,
    )
    return result, dm, net


def test_run_reid_requires_matching_seed(tmp_path):
    dm = _TinyReIDDataModule(split_seed=42)
    with pytest.raises(ValueError, match="seed"):
        run_reid(_tiny_net(dm), dm, seed=7, runs_dir=tmp_path)


def test_run_reid_artifacts(reid_result):
    result, dm, net = reid_result
    assert isinstance(result, ReIDResult)
    assert result.checkpoint_path.exists()
    assert result.log_dir.is_dir()
    for path in (
        result.manifest_path,
        result.predictions_path,
        result.summary_path,
    ):
        assert path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["seed"] == 42
    assert manifest["train"] == list(dm.split_manifest.train)
    assert manifest["dataset"] == "tiny_reid"


def test_run_reid_metrics_and_thresholds(reid_result):
    result, dm, net = reid_result
    expected = {
        "test/rank1",
        "test/rank3",
        "test/mAP",
        "test/auroc",
        "test/eer",
    }
    assert expected <= result.metrics.keys()
    for point in ("eer_threshold", "far05_threshold"):
        for metric in (
            "dir",
            "far",
            "unknown_rejection",
            "known_acceptance",
            "accepted_top1",
        ):
            assert f"test/{point}/{metric}" in result.metrics
    assert {
        "eer_threshold",
        "eer",
        "far_threshold",
        "far_achieved",
        "far_target",
    } <= result.thresholds.keys()
    summary = json.loads(result.summary_path.read_text())
    assert summary["thresholds"]["calibrated_on"] == "validation"
    assert summary["metrics"] == result.metrics


def test_run_reid_predictions_rows(reid_result):
    result, dm, net = reid_result
    with result.predictions_path.open() as f:
        rows = list(csv.DictReader(f))
    known = 3 * dm.samples
    unknown = 1 * dm.samples
    assert len(rows) == known + unknown
    assert {
        "probe_identity",
        "known",
        "top_identity",
        "top_score",
        "rank",
        "pred_eer_threshold",
        "pred_far05_threshold",
    } <= rows[0].keys()
    enrolled = set(dm.split_manifest.test_enrolled)
    for row in rows:
        assert row["pred_eer_threshold"] in enrolled | {"unknown"}


def test_run_reid_restores_best_weights(reid_result):
    result, dm, net = reid_result
    state = torch.load(
        result.checkpoint_path, map_location="cpu", weights_only=True
    )["state_dict"]
    for key, value in net.state_dict().items():
        assert torch.equal(value, state["net." + key])


def test_run_reid_thresholds_come_from_validation(reid_result):
    result, dm, net = reid_result
    loaders = dm.validation_loaders_by_role()
    gallery_z, gallery_y = _embed_loader(net, loaders["gallery"])
    known_z, _ = _embed_loader(net, loaders["known_probes"])
    unknown_z, _ = _embed_loader(net, loaders["unknown_probes"])
    scores = score_gallery_probe(
        gallery_z, gallery_y, torch.cat([known_z, unknown_z])
    )
    known_mask = torch.tensor(
        [True] * known_z.shape[0] + [False] * unknown_z.shape[0]
    )
    recomputed = calibrate_thresholds(scores.top_scores, known_mask)
    assert result.thresholds["eer_threshold"] == pytest.approx(
        recomputed["eer_threshold"]
    )
    assert result.thresholds["far_threshold"] == pytest.approx(
        recomputed["far_threshold"]
    )


# --- run_reid_repeats ---


def _run_repeats(tmp_path, seeds, **kwargs):
    calls = {"net": 0, "dm": 0}
    nets = []

    def dm_factory(seed):
        calls["dm"] += 1
        return _TinyReIDDataModule(split_seed=seed)

    def net_factory(dm):
        calls["net"] += 1
        net = _tiny_net(dm)
        nets.append(net)
        return net

    result = run_reid_repeats(
        net_factory,
        dm_factory,
        seeds=seeds,
        max_epochs=1,
        name="repeat-smoke",
        accelerator="cpu",
        runs_dir=tmp_path,
        **kwargs,
    )
    return result, calls, nets


@pytest.fixture(scope="module")
def repeats_result(tmp_path_factory):
    return _run_repeats(tmp_path_factory.mktemp("reid_repeats"), seeds=(42, 43))


def test_repeats_call_factories_once_per_seed(repeats_result):
    result, calls, nets = repeats_result
    assert calls == {"net": 2, "dm": 2}
    assert len({id(net) for net in nets}) == 2
    assert len(result.repeats) == 2


def test_repeats_manifests_match_seeds(repeats_result):
    result, _, _ = repeats_result
    names = [f"s{i:02d}" for i in range(1, 15)]
    for seed, repeat in zip((42, 43), result.repeats):
        manifest = json.loads(repeat.manifest_path.read_text())
        expected = make_identity_split(names, seed=seed)
        assert manifest["seed"] == seed
        assert manifest["train"] == list(expected.train)


def test_repeats_aggregate_mean_and_sample_std(repeats_result):
    import statistics

    result, _, _ = repeats_result
    for key, stats in result.aggregate_metrics.items():
        values = [r.metrics[key] for r in result.repeats]
        assert stats["mean"] == pytest.approx(statistics.fmean(values))
        assert stats["std"] == pytest.approx(statistics.stdev(values))


def test_repeats_single_seed_std_is_zero(tmp_path):
    result, _, _ = _run_repeats(tmp_path, seeds=(42,))
    for stats in result.aggregate_metrics.values():
        assert stats["std"] == 0.0


def test_repeats_summary_lists_artifacts(repeats_result):
    result, _, _ = repeats_result
    summary = json.loads(result.summary_path.read_text())
    assert summary["seeds"] == [42, 43]
    assert len(summary["repeats"]) == 2
    for seed, entry in zip((42, 43), summary["repeats"]):
        assert entry["seed"] == seed
        assert Path(entry["summary"]).exists()
        assert Path(entry["checkpoint"]).exists()
    # aggregate JSON parses with allow_nan-style strictness: reparse strictly
    strict = json.loads(
        result.summary_path.read_text(),
        parse_constant=lambda c: pytest.fail(f"non-finite value {c}"),
    )
    assert strict["metrics"].keys() == result.aggregate_metrics.keys()


def test_repeats_failure_carries_seed_context(tmp_path):
    def bad_dm_factory(seed):
        if seed == 43:
            raise RuntimeError("boom")
        return _TinyReIDDataModule(split_seed=seed)

    with pytest.raises(RuntimeError, match="43"):
        run_reid_repeats(
            _tiny_net,
            bad_dm_factory,
            seeds=(42, 43),
            max_epochs=1,
            name="fail",
            accelerator="cpu",
            runs_dir=tmp_path,
        )


def test_repeats_require_seeds(tmp_path):
    with pytest.raises(ValueError, match="seeds"):
        run_reid_repeats(
            _tiny_net, _TinyReIDDataModule, seeds=(), runs_dir=tmp_path
        )


# --- end-to-end on a generated NTU-Fi tree ---


def test_run_reid_end_to_end_generated_ntu(fake_ntu_root, tmp_path):
    from rfsensing import data

    dm = data.build(
        "ntu_fi_humanid_reid",
        root=fake_ntu_root,
        split_seed=42,
        identities_per_batch=3,
        samples_per_identity=2,
        eval_batch_size=4,
    )
    net = models.build(
        "resnet18",
        in_shape=dm.sample_shape,
        num_classes=dm.output_dim,
        base_width=8,
    )
    result = run_reid(
        net,
        dm,
        max_epochs=1,
        name="e2e",
        seed=42,
        accelerator="cpu",
        runs_dir=tmp_path,
    )
    assert result.checkpoint_path.exists()
    assert result.manifest_path.exists()
    summary = json.loads(result.summary_path.read_text())
    assert summary["thresholds"]["calibrated_on"] == "validation"
    with result.predictions_path.open() as f:
        rows = list(csv.DictReader(f))
    # 3 test-enrolled identities x 2 test samples + 1 unknown x 2 samples.
    assert len(rows) == 8
    manifest = json.loads(result.manifest_path.read_text())
    assert sorted(
        manifest["train"]
        + manifest["val_enrolled"]
        + manifest["val_unknown"]
        + manifest["test_enrolled"]
        + manifest["test_unknown"]
    ) == dm.identity_names
    for key in ("test/mAP", "test/auroc", "test/eer_threshold/dir"):
        assert key in result.metrics
        assert 0.0 <= result.metrics[key] <= 1.0


# --- device handling ---


def test_embed_loader_accepts_device():
    dm = _TinyReIDDataModule()
    dm.setup()
    net = _tiny_net(dm)
    loader = dm.test_loaders_by_role()["gallery"]
    embeddings, labels = _embed_loader(net, loader, device="cpu")
    assert embeddings.device.type == "cpu"
    assert labels.device.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
def test_run_reid_on_mps(tmp_path):
    dm = _TinyReIDDataModule(split_seed=42)
    net = _tiny_net(dm)
    result = run_reid(
        net,
        dm,
        max_epochs=1,
        name="mps-smoke",
        seed=42,
        accelerator="mps",
        runs_dir=tmp_path,
    )
    assert "test/mAP" in result.metrics
    assert result.summary_path.exists()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
def test_run_reid_device_override(tmp_path):
    dm = _TinyReIDDataModule(split_seed=42)
    net = _tiny_net(dm)
    result = run_reid(
        net,
        dm,
        max_epochs=1,
        name="device-override",
        seed=42,
        accelerator="mps",
        device="cpu",
        runs_dir=tmp_path,
    )
    assert "test/mAP" in result.metrics
    # Evaluation ran on the requested device: the net ends up there.
    assert next(net.parameters()).device.type == "cpu"


# --- objective and detection-score options ---


def test_run_reid_rejects_invalid_options(tmp_path):
    dm = _TinyReIDDataModule(split_seed=42)
    with pytest.raises(ValueError, match="detection_score"):
        run_reid(
            _tiny_net(dm), dm, seed=42, runs_dir=tmp_path,
            detection_score="softmax",
        )
    with pytest.raises(ValueError, match="objective"):
        run_reid(
            _tiny_net(dm), dm, seed=42, runs_dir=tmp_path, objective="arcface"
        )


@pytest.fixture(scope="module")
def gap_supcon_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("reid_gap")
    dm = _TinyReIDDataModule(split_seed=42)
    net = _tiny_net(dm)
    result = run_reid(
        net,
        dm,
        max_epochs=1,
        name="gap-supcon",
        seed=42,
        accelerator="cpu",
        runs_dir=tmp_path,
        objective="supcon",
        detection_score="top_gap",
    )
    return result, dm, net


def test_run_reid_gap_supcon_smoke(gap_supcon_result):
    result, dm, net = gap_supcon_result
    assert "test/mAP" in result.metrics
    assert "test/eer_threshold/dir" in result.metrics
    summary = json.loads(result.summary_path.read_text())
    assert summary["detection_score"] == "top_gap"
    config = json.loads((result.log_dir / "config.json").read_text())
    assert config["objective"] == "supcon"
    assert config["detection_score"] == "top_gap"


def test_run_reid_gap_thresholds_calibrated_on_gap_scores(gap_supcon_result):
    result, dm, net = gap_supcon_result
    loaders = dm.validation_loaders_by_role()
    gallery_z, gallery_y = _embed_loader(net, loaders["gallery"])
    known_z, _ = _embed_loader(net, loaders["known_probes"])
    unknown_z, _ = _embed_loader(net, loaders["unknown_probes"])
    scores = score_gallery_probe(
        gallery_z, gallery_y, torch.cat([known_z, unknown_z])
    )
    known_mask = torch.tensor(
        [True] * known_z.shape[0] + [False] * unknown_z.shape[0]
    )
    recomputed = calibrate_thresholds(scores.top_gaps, known_mask)
    assert result.thresholds["eer_threshold"] == pytest.approx(
        recomputed["eer_threshold"]
    )
    assert result.thresholds["far_threshold"] == pytest.approx(
        recomputed["far_threshold"]
    )


def test_run_reid_predictions_include_detection_score(gap_supcon_result):
    result, dm, net = gap_supcon_result
    with result.predictions_path.open() as f:
        rows = list(csv.DictReader(f))
    assert "detection_score" in rows[0]
    for row in rows:
        detection = float(row["detection_score"])
        expected = (
            row["top_identity"]
            if detection >= result.thresholds["eer_threshold"]
            else "unknown"
        )
        assert row["pred_eer_threshold"] == expected
