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


def test_lstm_seq_axis():
    net = models.build("lstm", in_shape=(22, 20, 20), num_classes=6, seq_axis=0)
    out = net(torch.randn(2, 22, 20, 20))
    assert out.shape == (2, 6)