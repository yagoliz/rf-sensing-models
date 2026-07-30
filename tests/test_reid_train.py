import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rfsensing import models
from rfsensing.train.reid import ReIDModule, batch_hard_triplet_loss


# --- batch_hard_triplet_loss ---


def _unit(*vectors):
    return F.normalize(torch.tensor(vectors, dtype=torch.float32), dim=1)


def test_triplet_loss_uses_hardest_positive_and_negative():
    # Anchor 0 (label 0): positives at 0 and 60 deg, negative at 90 deg.
    embeddings = _unit(
        [1.0, 0.0],
        [math.cos(math.radians(60)), math.sin(math.radians(60))],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    )
    labels = torch.tensor([0, 0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)
    # Anchor 0: hardest positive d=0.5 (60 deg), hardest negative d=1.0
    #   -> relu(0.5 - 1.0 + 0.3) = 0
    # Anchor 1: hardest positive d=0.5, hardest negative 1 - cos(30)=0.134
    #   -> relu(0.5 - 0.134 + 0.3) = 0.666
    # Anchor 2: same as anchor 0 -> 0
    # Anchor 3: positive d=2 (opposite), negative d=1-cos(30)
    #   -> relu(2 - 0.134 + 0.3) = 2.166
    # Anchor 4: positive d=2, hardest negative d=1 ([1,0] and [0,1]... )
    per_anchor = []
    z = embeddings
    d = 1.0 - z @ z.T
    for i in range(5):
        pos = [j for j in range(5) if labels[j] == labels[i] and j != i]
        neg = [j for j in range(5) if labels[j] != labels[i]]
        hardest_pos = max(d[i, j].item() for j in pos)
        hardest_neg = min(d[i, j].item() for j in neg)
        per_anchor.append(max(0.0, hardest_pos - hardest_neg + 0.3))
    assert loss.item() == pytest.approx(sum(per_anchor) / 5, abs=1e-5)


def test_triplet_loss_zero_when_margin_satisfied():
    embeddings = _unit([1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0])
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)
    assert loss.item() == pytest.approx(0.0)


def test_triplet_loss_propagates_gradients():
    raw = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = batch_hard_triplet_loss(F.normalize(raw, dim=1), labels, margin=1.0)
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_triplet_loss_rejects_invalid_batches():
    embeddings = _unit([1.0, 0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        batch_hard_triplet_loss(embeddings, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="negative"):
        batch_hard_triplet_loss(embeddings, torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="margin"):
        batch_hard_triplet_loss(
            embeddings, torch.tensor([0, 0]), margin=0.0
        )


# --- ReIDModule ---

IN_SHAPE = (3, 16, 20)


def _net(num_classes=4, name="mlp", **kwargs):
    if name == "mlp":
        kwargs.setdefault("hidden_dims", (16,))
    return models.build(
        name, in_shape=IN_SHAPE, num_classes=num_classes, **kwargs
    )


def _logged(module, monkeypatch):
    logged = {}
    monkeypatch.setattr(
        module, "log", lambda name, value, **kw: logged.__setitem__(name, value)
    )
    return logged


def test_module_single_encoder_pass(monkeypatch):
    net = _net()
    module = ReIDModule(net, num_train_identities=4)
    calls = {"embed": 0}
    original = net.embed

    def counting_embed(x):
        calls["embed"] += 1
        return original(x)

    monkeypatch.setattr(net, "embed", counting_embed)
    monkeypatch.setattr(module, "log", lambda *a, **k: None)
    batch = (torch.randn(8, *IN_SHAPE), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    module.training_step(batch, 0)
    assert calls["embed"] == 1


def test_module_logs_all_train_losses(monkeypatch):
    module = ReIDModule(_net(), num_train_identities=4)
    logged = _logged(module, monkeypatch)
    batch = (torch.randn(8, *IN_SHAPE), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    loss = module.training_step(batch, 0)
    assert {"train/loss", "train/ce_loss", "train/triplet_loss"} <= logged.keys()
    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(
        logged["train/ce_loss"].item()
        + module.hparams.triplet_weight * logged["train/triplet_loss"].item()
    )


def test_module_constructor_validation():
    with pytest.raises(ValueError, match="embed"):
        ReIDModule(nn.Linear(4, 4), num_train_identities=4)
    with pytest.raises(ValueError, match="head"):
        bad = nn.Module()
        bad.embed = lambda x: x
        ReIDModule(bad, num_train_identities=4)
    with pytest.raises(ValueError, match="classifier"):
        ReIDModule(_net(num_classes=5), num_train_identities=4)
    with pytest.raises(ValueError, match="triplet_weight"):
        ReIDModule(_net(), num_train_identities=4, triplet_weight=-1.0)


def test_module_validation_epoch_computes_retrieval(monkeypatch):
    net = _net()
    module = ReIDModule(net, num_train_identities=4)
    logged = _logged(module, monkeypatch)
    gallery_batch = (torch.randn(6, *IN_SHAPE), torch.tensor([3, 3, 7, 7, 9, 9]))
    probe_batch = (torch.randn(6, *IN_SHAPE), torch.tensor([3, 7, 9, 3, 7, 9]))
    module.validation_step(gallery_batch, 0, 0)
    module.validation_step(probe_batch, 0, 1)
    module.on_validation_epoch_end()
    assert {"val/mAP", "val/rank1", "val/rank3"} <= logged.keys()
    assert 0.0 <= logged["val/mAP"] <= 1.0
    # Buffers must be cleared for the next epoch.
    assert not module._val_embeddings[0] and not module._val_embeddings[1]


def test_module_set_epoch_reaches_sampler():
    from torch.utils.data import DataLoader, TensorDataset

    import lightning as L

    from rfsensing.data.reid import IdentityBatchSampler

    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    dataset = TensorDataset(torch.randn(12, *IN_SHAPE), labels)
    sampler = IdentityBatchSampler(labels.tolist(), 2, 2, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler)
    module = ReIDModule(_net(), num_train_identities=4)
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=loader)
    assert sampler._epoch == 1


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("resnet18", {"base_width": 8}),
        ("vit", {"patch_size": 4, "embed_dim": 16, "depth": 1, "num_heads": 2}),
    ],
)
def test_module_supports_reference_encoders(name, kwargs, monkeypatch):
    net = _net(name=name, **kwargs)
    module = ReIDModule(net, num_train_identities=4)
    logged = _logged(module, monkeypatch)
    x = torch.randn(8, *IN_SHAPE)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    z = F.normalize(net.embed(x), dim=1)
    assert z.shape[0] == 8
    assert torch.allclose(
        z.norm(dim=1), torch.ones(8), atol=1e-5
    )
    loss = module.training_step((x, y), 0)
    assert torch.isfinite(loss)
    module.validation_step((x[:4], torch.tensor([3, 3, 7, 7])), 0, 0)
    module.validation_step((x[4:], torch.tensor([3, 7, 3, 7])), 0, 1)
    module.on_validation_epoch_end()
    assert "val/mAP" in logged


def test_vit_one_step_contract_on_generated_ntu(fake_ntu_root, monkeypatch):
    from rfsensing import data

    dm = data.build(
        "ntu_fi_humanid_reid",
        root=fake_ntu_root,
        split_seed=42,
        identities_per_batch=2,
        samples_per_identity=2,
        eval_batch_size=4,
    )
    dm.setup("fit")
    net = models.build(
        "vit",
        in_shape=dm.sample_shape,
        num_classes=dm.output_dim,
        patch_size=(38, 100),
        embed_dim=16,
        depth=1,
        num_heads=2,
    )
    module = ReIDModule(net, num_train_identities=dm.output_dim)
    logged = _logged(module, monkeypatch)
    x, y = next(iter(dm.train_dataloader()))
    loss = module.training_step((x, y), 0)
    assert torch.isfinite(loss)
    gallery, probes = dm.val_dataloader()
    module.validation_step(next(iter(gallery)), 0, 0)
    module.validation_step(next(iter(probes)), 0, 1)
    module.on_validation_epoch_end()
    assert {"val/mAP", "val/rank1", "val/rank3"} <= logged.keys()


needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


@needs_mps
def test_module_validation_works_on_mps(monkeypatch):
    net = _net().to("mps")
    module = ReIDModule(net, num_train_identities=4)
    logged = _logged(module, monkeypatch)
    gallery = (
        torch.randn(4, *IN_SHAPE, device="mps"),
        torch.tensor([3, 3, 7, 7], device="mps"),
    )
    probes = (
        torch.randn(4, *IN_SHAPE, device="mps"),
        torch.tensor([3, 7, 3, 7], device="mps"),
    )
    module.validation_step(gallery, 0, 0)
    # Buffers must be device-agnostic so epoch-end metrics never mix devices.
    assert module._val_embeddings[0][0].device.type == "cpu"
    module.validation_step(probes, 0, 1)
    module.on_validation_epoch_end()
    assert {"val/mAP", "val/rank1", "val/rank3"} <= logged.keys()


# --- supcon_loss ---


def test_supcon_loss_hand_calculated():
    from rfsensing.train.reid import supcon_loss

    embeddings = _unit([1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0])
    labels = torch.tensor([0, 0, 1, 1])
    loss = supcon_loss(embeddings, labels, temperature=1.0)
    # Every anchor: one positive at sim 1, two negatives at sim 0
    #   -> -log(e / (e + 2)) each.
    expected = math.log(math.e + 2) - 1.0
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_supcon_loss_rewards_separation():
    from rfsensing.train.reid import supcon_loss

    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    separated = F.normalize(
        torch.eye(4).repeat_interleave(2, dim=0)
        + 0.01 * torch.randn(8, 4, generator=torch.Generator().manual_seed(0)),
        dim=1,
    )
    collapsed = F.normalize(torch.ones(8, 4), dim=1)
    assert supcon_loss(separated, labels, temperature=0.5) < supcon_loss(
        collapsed, labels, temperature=0.5
    )


def test_supcon_loss_propagates_gradients():
    from rfsensing.train.reid import supcon_loss

    raw = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supcon_loss(F.normalize(raw, dim=1), labels)
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_supcon_loss_rejects_invalid_batches():
    from rfsensing.train.reid import supcon_loss

    embeddings = _unit([1.0, 0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        supcon_loss(embeddings, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="negative"):
        supcon_loss(embeddings, torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="temperature"):
        supcon_loss(embeddings, torch.tensor([0, 0]), temperature=0.0)


# --- ReIDModule objective selection ---


def test_module_supcon_objective(monkeypatch):
    module = ReIDModule(
        _net(), num_train_identities=4, objective="supcon", triplet_weight=2.0
    )
    logged = _logged(module, monkeypatch)
    batch = (torch.randn(8, *IN_SHAPE), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    loss = module.training_step(batch, 0)
    assert "train/supcon_loss" in logged
    assert "train/triplet_loss" not in logged
    assert loss.item() == pytest.approx(
        logged["train/ce_loss"].item() + 2.0 * logged["train/supcon_loss"].item()
    )


def test_module_rejects_invalid_objective():
    with pytest.raises(ValueError, match="objective"):
        ReIDModule(_net(), num_train_identities=4, objective="arcface")
    with pytest.raises(ValueError, match="temperature"):
        ReIDModule(
            _net(), num_train_identities=4, objective="supcon",
            supcon_temperature=0.0,
        )
