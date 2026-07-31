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
    # seq_len=10 keeps whofi valid on the shortest benchmark time axis.
    ("whofi", {"seq_len": 10, "d_model": 32, "num_heads": 4, "embed_dim": 16}),
    ("whofi", {"encoder": "lstm", "seq_len": 10, "d_model": 32, "embed_dim": 16}),
    ("whofi", {"encoder": "bilstm", "seq_len": 10, "d_model": 32, "embed_dim": 16}),
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


def test_whofi_default_matches_ntu_fi_shape():
    net = models.build("whofi", in_shape=(3, 114, 500), num_classes=14)
    z = net.embed(torch.randn(2, 3, 114, 500))
    assert z.shape == (2, 128)
    # Default sequence: 500 packets uniformly subsampled to 100.
    assert net.step_indices.numel() == 100
    assert net.step_indices[0] == 0 and net.step_indices[-1] == 499


def test_whofi_rejects_invalid_options():
    with pytest.raises(ValueError, match="encoder"):
        models.build(
            "whofi", in_shape=(3, 114, 500), num_classes=14, encoder="gru"
        )
    with pytest.raises(ValueError, match="seq_len"):
        models.build(
            "whofi", in_shape=(22, 20, 20), num_classes=6, seq_len=100
        )


def test_whofi_full_sequence_when_seq_len_none():
    net = models.build(
        "whofi",
        in_shape=(22, 20, 20),
        num_classes=6,
        seq_len=None,
        d_model=16,
        num_heads=2,
        embed_dim=8,
    )
    assert net.step_indices is None
    assert net.positional_encoding.shape == (20, 16)
    assert net(torch.randn(2, 22, 20, 20)).shape == (2, 6)


def test_whofi_bilstm_readout_concatenates_directions():
    net = models.build(
        "whofi",
        in_shape=(3, 114, 500),
        num_classes=14,
        encoder="bilstm",
        d_model=32,
        embed_dim=16,
    )
    assert net.signature.in_features == 64  # forward + backward hidden states
    assert net.embed(torch.randn(2, 3, 114, 500)).shape == (2, 16)