import json

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rfsensing import models
from rfsensing.train.whofi_run import (
    leave_one_out_metrics,
    run_whofi,
    run_whofi_repeats,
)

IN_SHAPE = (3, 16, 20)


# --- leave_one_out_metrics ---


def _unit(*vectors):
    return F.normalize(torch.tensor(vectors, dtype=torch.float32), dim=1)


def test_loo_metrics_perfect_clusters():
    embeddings = _unit([1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0])
    labels = torch.tensor([0, 0, 1, 1])
    metrics = leave_one_out_metrics(embeddings, labels, ranks=(1, 2))
    assert metrics["rank1"] == pytest.approx(1.0)
    assert metrics["rank2"] == pytest.approx(1.0)
    assert metrics["mAP"] == pytest.approx(1.0)


def test_loo_metrics_never_match_self():
    # Each sample's nearest neighbour excluding itself is the other identity:
    # sample 1 sits between its twin and identity 1's cluster.
    embeddings = _unit([1.0, 0.0], [0.6, 0.8], [0.5, 0.9], [0.4, 1.0])
    labels = torch.tensor([0, 0, 1, 1])
    metrics = leave_one_out_metrics(embeddings, labels, ranks=(1,))
    # Sample at [0.6, 0.8] scores identity 1 (cos ~0.99) above its distant
    # twin (cos 0.6): with self excluded, rank-1 must drop below 1.
    assert metrics["rank1"] < 1.0


def test_loo_metrics_validation():
    embeddings = _unit([1.0, 0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="at least two samples"):
        leave_one_out_metrics(embeddings, torch.tensor([0, 1]), ranks=(1,))
    with pytest.raises(ValueError, match="rank"):
        leave_one_out_metrics(
            _unit([1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]),
            torch.tensor([0, 0, 1, 1]),
            ranks=(5,),
        )
    with pytest.raises(ValueError, match="labels"):
        leave_one_out_metrics(embeddings, torch.tensor([0]), ranks=(1,))


# --- run_whofi ---


class _LabeledDataset(Dataset):
    def __init__(self, num_identities, per_identity, seed):
        generator = torch.Generator().manual_seed(seed)
        count = num_identities * per_identity
        self.x = torch.randn(count, *IN_SHAPE, generator=generator)
        self.labels = [i for i in range(num_identities) for _ in range(per_identity)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.x[i], self.labels[i]


class _TinyWhoFiDataModule:
    name = "tiny_whofi"
    batch_size = 8

    def setup(self, stage=None):
        self.train_set = _LabeledDataset(4, 6, seed=0)
        self.test_set = _LabeledDataset(4, 4, seed=1)


def _tiny_net():
    return models.build(
        "whofi",
        in_shape=IN_SHAPE,
        num_classes=4,
        seq_len=10,
        d_model=16,
        num_heads=2,
        ff_dim=16,
        embed_dim=8,
    )


@pytest.fixture(scope="module")
def whofi_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("whofi")
    result = run_whofi(
        _tiny_net(),
        _TinyWhoFiDataModule(),
        max_epochs=2,
        seed=42,
        identities_per_batch=4,
        ranks=(1, 3),
        device="cpu",
        runs_dir=tmp_path,
    )
    return result


def test_run_whofi_smoke(whofi_result):
    for protocol in ("enrollment", "loo"):
        for metric in ("rank1", "rank3", "mAP"):
            value = whofi_result.metrics[f"test/{protocol}/{metric}"]
            assert 0.0 <= value <= 1.0
    assert whofi_result.checkpoint_path.exists()
    summary = json.loads(whofi_result.summary_path.read_text())
    assert summary["metrics"] == whofi_result.metrics
    config = json.loads((whofi_result.log_dir / "config.json").read_text())
    assert config["objective"] == "inbatch"
    assert config["max_epochs"] == 2


def test_run_whofi_checkpoint_restores(whofi_result):
    net = _tiny_net()
    net.load_state_dict(
        torch.load(whofi_result.checkpoint_path, weights_only=True)
    )


def test_run_whofi_rejects_bad_epochs(tmp_path):
    with pytest.raises(ValueError, match="max_epochs"):
        run_whofi(
            _tiny_net(),
            _TinyWhoFiDataModule(),
            max_epochs=0,
            device="cpu",
            runs_dir=tmp_path,
        )


def test_run_whofi_repeats_aggregates(tmp_path):
    result = run_whofi_repeats(
        lambda dm: _tiny_net(),
        lambda seed: _TinyWhoFiDataModule(),
        seeds=(42, 43),
        max_epochs=1,
        identities_per_batch=4,
        ranks=(1,),
        device="cpu",
        name="tiny-whofi",
        runs_dir=tmp_path,
    )
    assert len(result.repeats) == 2
    stats = result.aggregate_metrics["test/enrollment/rank1"]
    assert set(stats) == {"mean", "std"}
    saved = json.loads(result.summary_path.read_text())
    assert saved["seeds"] == [42, 43]
    assert len(saved["repeats"]) == 2


def test_run_whofi_repeats_carries_seed_context(tmp_path):
    with pytest.raises(RuntimeError, match="seed 42"):
        run_whofi_repeats(
            lambda dm: _tiny_net(),
            lambda seed: _TinyWhoFiDataModule(),
            seeds=(42,),
            max_epochs=0,  # invalid, fails inside the repeat
            device="cpu",
            runs_dir=tmp_path,
        )
