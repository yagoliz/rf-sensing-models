import torch

from rfsensing import models
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