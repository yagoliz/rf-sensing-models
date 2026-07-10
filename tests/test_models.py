import pytest
import torch

from rfsensing import models

# One shape per benchmark dataset: UT-HAR, NTU-Fi, Widar.
SHAPES = [(1, 250, 90), (3, 114, 500), (22, 20, 20)]

# (registered name, extra kwargs) — extended as models are added.
CASES = [
    ("mlp", {}),
    ("lenet", {}),
    ("lstm", {}),
    ("lstm", {"bidirectional": True}),
    ("resnet18", {}),
    ("vit", {}),
]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("name,kwargs", CASES, ids=lambda c: str(c))
def test_forward_shape(name, kwargs, shape):
    net = models.build(name, in_shape=shape, num_classes=6, **kwargs)
    x = torch.randn(2, *shape)
    out = net(x)
    assert out.shape == (2, 6)


def test_unknown_model_lists_available():
    with pytest.raises(KeyError, match="Unknown model 'nope'"):
        models.build("nope")


def test_list_available_contains_mlp():
    assert "mlp" in models.list_available()


def test_vit_rejects_oversized_patch():
    with pytest.raises(ValueError, match="patch_size"):
        models.build("vit", in_shape=(1, 8, 8), num_classes=4, patch_size=10)


@pytest.mark.parametrize("name,kwargs", CASES, ids=lambda c: str(c))
def test_embed_head_contract(name, kwargs):
    net = models.build(name, in_shape=(1, 250, 90), num_classes=6, **kwargs)
    net.eval()
    x = torch.randn(2, 1, 250, 90)
    z = net.embed(x)
    assert z.ndim == 2 and z.shape[0] == 2
    assert torch.allclose(net(x), net.head(z), atol=1e-6)


def test_lstm_seq_axis():
    net = models.build("lstm", in_shape=(22, 20, 20), num_classes=6, seq_axis=0)
    out = net(torch.randn(2, 22, 20, 20))
    assert out.shape == (2, 6)