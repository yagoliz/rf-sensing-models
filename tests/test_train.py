import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from rfsensing import data, models, train
from rfsensing.data.base import CSIDataModule
from rfsensing.train.module import ClassificationModule, RegressionModule


def _module():
    net = models.build("mlp", in_shape=(3, 8, 8), num_classes=4, hidden_dims=(16,))
    return ClassificationModule(net, num_classes=4)


def test_forward_delegates_to_net():
    module = _module()
    out = module(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 4)


def test_training_step_returns_scalar_loss():
    module = _module()
    batch = (torch.randn(8, 3, 8, 8), torch.randint(0, 4, (8,)))
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_run_end_to_end_synthetic(tmp_path):
    dm = data.build("synthetic", batch_size=8)
    net = models.build(
        "mlp", in_shape=dm.sample_shape, num_classes=dm.num_classes, hidden_dims=(16,)
    )
    result = train.run(
        net, dm, max_epochs=2, name="smoke", runs_dir=tmp_path, accelerator="cpu"
    )
    assert "test/acc" in result.metrics
    assert 0.0 <= result.metrics["test/acc"] <= 1.0
    assert result.checkpoint_path.exists()
    assert result.log_dir.is_dir()
    assert str(tmp_path) in str(result.log_dir)


def test_load_best_net_restores_checkpoint_weights(tmp_path):
    dm = data.build("synthetic", batch_size=8)
    net = models.build(
        "mlp", in_shape=dm.sample_shape, num_classes=dm.num_classes, hidden_dims=(16,)
    )
    result = train.run(
        net, dm, max_epochs=2, name="smoke", runs_dir=tmp_path, accelerator="cpu"
    )
    restored = train.load_best_net(net, result)
    assert restored is net
    ckpt = torch.load(
        result.checkpoint_path, map_location="cpu", weights_only=True
    )["state_dict"]
    for key, value in restored.state_dict().items():
        assert torch.equal(value, ckpt["net." + key])


class _FixedNet(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = nn.Parameter(torch.as_tensor(outputs, dtype=torch.float32))

    def forward(self, x):
        return self.outputs[: x.shape[0]]


def test_ordered_classification_tracks_count_metrics(monkeypatch):
    net = _FixedNet([[5, 0, 0], [0, 0, 5], [0, 0, 5]])
    module = ClassificationModule(
        net, num_classes=3, ordered_values=(0.0, 1.0, 2.0)
    )
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = (torch.randn(3, 1), torch.tensor([0, 1, 2]))
    module.test_step(batch, 0)
    assert module.test_mae.compute().item() == pytest.approx(1 / 3)
    assert module.test_within_1.compute().item() == pytest.approx(1.0)


def test_regression_tracks_raw_and_rounded_metrics(monkeypatch):
    module = RegressionModule(
        _FixedNet([[-1.0], [1.4], [7.0]]), target_range=(0.0, 5.0)
    )
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = (torch.randn(3, 1), torch.tensor([0.0, 2.0, 5.0]))
    loss = module.test_step(batch, 0)
    assert loss is None
    assert module.test_mae.compute().item() == pytest.approx(1.2)
    assert module.test_within_1.compute().item() == pytest.approx(2 / 3)
    assert module.test_rounded_acc.compute().item() == pytest.approx(2 / 3)


def test_regression_training_step_returns_scalar_loss(monkeypatch):
    module = RegressionModule(_FixedNet([[1.0], [2.0]]))
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    loss = module.training_step(
        (torch.randn(2, 1), torch.tensor([1.5, 1.5])), 0
    )
    assert loss.ndim == 0
    assert loss.requires_grad


class _SyntheticRegressionDataModule(CSIDataModule):
    name = "synthetic_regression"
    sample_shape = (1, 4, 4)
    class_names = []
    task_type = "regression"
    target_range = (0.0, 5.0)
    checkpoint_monitor = "val/mae"
    checkpoint_mode = "min"

    @property
    def output_dim(self):
        return 1

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(48, *self.sample_shape, generator=generator)
        y = (x.mean(dim=(1, 2, 3)) + 2.5).clamp(0, 5)
        dataset = TensorDataset(x, y)
        self.train_set = dataset
        self.val_set = dataset
        self.test_set = dataset

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)


def test_run_end_to_end_regression(tmp_path):
    dm = _SyntheticRegressionDataModule(batch_size=8)
    net = models.build(
        "mlp", in_shape=dm.sample_shape, num_classes=dm.output_dim, hidden_dims=(16,)
    )
    result = train.run(
        net,
        dm,
        max_epochs=2,
        name="regression-smoke",
        runs_dir=tmp_path,
        accelerator="cpu",
    )
    assert {"test/mae", "test/within_1", "test/rounded_acc"} <= result.metrics.keys()
    assert result.metrics["test/mae"] >= 0
    assert result.checkpoint_path.exists()
    assert train.load_best_net(net, result) is net
