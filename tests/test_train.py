import torch

from rfsensing import data, models, train
from rfsensing.train.module import ClassificationModule


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