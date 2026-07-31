"""WhoFi closed-set reproduction runner (Avola et al. 2025, arXiv:2507.12869).

Faithful protocol on NTU-Fi HumanID: the encoder trains on the full training
split (all 14 identities) with the in-batch negative loss over paired
same-identity samples, Adam at lr 1e-4 with a StepLR decay of 0.95 every 50
epochs, and a fixed epoch budget with **no model selection** — the paper
reports the final model, validating hyperparameters by cross-validation
rather than by checkpoint picking.

The paper does not state how the evaluation gallery is built, so both
readings are computed:

- ``enrollment``: gallery = training samples, queries = test samples;
- ``loo``: leave-one-out inside the test split — every test sample queries
  all remaining test samples.
"""

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from rfsensing.data.base import CSIDataModule
from rfsensing.data.reid import IdentityBatchSampler
from rfsensing.eval.reid import retrieval_metrics, score_gallery_probe
from rfsensing.train.reid import ReIDModule, in_batch_negative_loss
from rfsensing.train.reid_run import _embed_loader, _save_json


@dataclass
class WhoFiResult:
    metrics: dict[str, float]
    log_dir: Path
    checkpoint_path: Path
    summary_path: Path


def leave_one_out_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    ranks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Rank-k and sample-level mAP with each sample querying all others."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("need at least two 2-D embeddings")
    labels = torch.as_tensor(labels).long().reshape(-1)
    if labels.numel() != embeddings.shape[0]:
        raise ValueError("labels must match embeddings")
    z = F.normalize(embeddings.float(), dim=1)
    similarities = z @ z.T
    similarities.fill_diagonal_(-torch.inf)
    identities = labels.unique()  # sorted ascending
    for k in ranks:
        if not 1 <= k <= identities.numel():
            raise ValueError(
                f"rank {k} is invalid for {identities.numel()} identities"
            )
    identity_scores = torch.stack(
        [similarities[:, labels == identity].amax(dim=1) for identity in identities],
        dim=1,
    )
    order = identity_scores.argsort(dim=1, descending=True, stable=True)
    ranked = identities[order]
    metrics = {
        f"rank{k}": (ranked[:, :k] == labels.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
        for k in ranks
    }
    sample_order = similarities.argsort(dim=1, descending=True, stable=True)
    relevant = labels.unsqueeze(0) == labels.unsqueeze(1)
    relevant &= ~torch.eye(labels.numel(), dtype=torch.bool)
    if not relevant.any(dim=1).all():
        raise ValueError("every identity needs at least two samples")
    relevant = relevant.gather(1, sample_order).float()
    positions = torch.arange(1, labels.numel() + 1, dtype=torch.float32)
    precision = relevant.cumsum(dim=1) / positions
    average_precision = (precision * relevant).sum(dim=1) / relevant.sum(dim=1)
    metrics["mAP"] = average_precision.mean().item()
    return metrics


def _resolve_device(device) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_whofi(
    net: nn.Module,
    dm: CSIDataModule,
    *,
    max_epochs: int = 300,
    seed: int = 42,
    lr: float = 1e-4,
    scheduler_step: int = 50,
    scheduler_gamma: float = 0.95,
    identities_per_batch: int = 8,
    ranks: tuple[int, ...] = (1, 3, 5),
    name: str | None = None,
    device: str | torch.device | None = None,
    runs_dir: str | Path = "runs",
) -> WhoFiResult:
    """Train one WhoFi repeat on a closed-set DataModule and evaluate it.

    ``dm`` must expose ``train_set``/``test_set`` datasets whose samples are
    ``(x, identity_index)`` with a ``labels`` list (the NTU-Fi contract).
    Batches pair two samples per identity for the in-batch negative loss.
    """
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
    L.seed_everything(seed, workers=True)
    dm.setup()
    device = _resolve_device(device)
    net = net.to(device)
    sampler = IdentityBatchSampler(
        dm.train_set.labels, identities_per_batch, 2, seed=seed
    )
    loader = DataLoader(dm.train_set, batch_sampler=sampler)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )
    net.train()
    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            raw = net.embed(x.to(device))
            query, gallery = ReIDModule._paired_views(raw, y.to(device))
            loss = in_batch_negative_loss(query, gallery)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    if not torch.isfinite(loss):
        raise RuntimeError("training diverged: final loss is not finite")

    eval_batch = getattr(dm, "batch_size", 64)
    train_z, train_y = _embed_loader(
        net, DataLoader(dm.train_set, batch_size=eval_batch), device
    )
    test_z, test_y = _embed_loader(
        net, DataLoader(dm.test_set, batch_size=eval_batch), device
    )
    scores = score_gallery_probe(train_z, train_y, test_z)
    enrollment = retrieval_metrics(
        scores, test_y, torch.ones(test_y.numel(), dtype=torch.bool), ranks=ranks
    )
    loo = leave_one_out_metrics(test_z, test_y, ranks=ranks)
    metrics = {
        **{f"test/enrollment/{k}": v for k, v in enrollment.items()},
        **{f"test/loo/{k}": v for k, v in loo.items()},
    }

    experiment = name or type(net).__name__.lower()
    seed_root = Path(runs_dir) / dm.name / experiment / f"seed{seed}"
    log_dir = seed_root / f"version_{len(list(seed_root.glob('version_*')))}"
    log_dir.mkdir(parents=True)
    checkpoint_path = log_dir / "checkpoint.pt"
    torch.save(net.state_dict(), checkpoint_path)
    _save_json(
        log_dir / "config.json",
        {
            "net": type(net).__name__,
            "max_epochs": max_epochs,
            "seed": seed,
            "lr": lr,
            "scheduler_step": scheduler_step,
            "scheduler_gamma": scheduler_gamma,
            "identities_per_batch": identities_per_batch,
            "objective": "inbatch",
        },
    )
    summary_path = _save_json(
        log_dir / "summary.json",
        {"metrics": metrics, "checkpoint": str(checkpoint_path)},
    )
    return WhoFiResult(
        metrics=metrics,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
    )


@dataclass
class RepeatedWhoFiResult:
    repeats: list[WhoFiResult]
    aggregate_metrics: dict[str, dict[str, float]]
    summary_path: Path


def run_whofi_repeats(
    net_factory: Callable[[CSIDataModule], nn.Module],
    datamodule_factory: Callable[[int], CSIDataModule],
    *,
    seeds: Sequence[int],
    max_epochs: int = 300,
    name: str | None = None,
    runs_dir: str | Path = "runs",
    **run_kwargs,
) -> RepeatedWhoFiResult:
    """One WhoFi repeat per seed, aggregated as mean ± sample std."""
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    repeats: list[WhoFiResult] = []
    experiment = name
    dataset_name = None
    for seed in seeds:
        try:
            dm = datamodule_factory(seed)
            net = net_factory(dm)
            if experiment is None:
                experiment = type(net).__name__.lower()
            dataset_name = dm.name
            result = run_whofi(
                net,
                dm,
                max_epochs=max_epochs,
                seed=seed,
                name=experiment,
                runs_dir=runs_dir,
                **run_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                f"whofi repeat for seed {seed} failed: {error}"
            ) from error
        repeats.append(result)
    aggregate_metrics = {
        key: {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for key in repeats[0].metrics
        for values in [[r.metrics[key] for r in repeats]]
    }
    parent = Path(runs_dir) / dataset_name / experiment
    parent.mkdir(parents=True, exist_ok=True)
    summary_path = _save_json(
        parent / "aggregate_summary.json",
        {
            "seeds": list(seeds),
            "metrics": aggregate_metrics,
            "repeats": [
                {
                    "seed": seed,
                    "log_dir": str(result.log_dir),
                    "checkpoint": str(result.checkpoint_path),
                    "summary": str(result.summary_path),
                }
                for seed, result in zip(seeds, repeats)
            ],
        },
    )
    return RepeatedWhoFiResult(
        repeats=repeats,
        aggregate_metrics=aggregate_metrics,
        summary_path=summary_path,
    )
