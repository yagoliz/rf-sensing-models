"""Open-set Re-ID experiment runners and artifact serialization.

``run_reid`` trains one repeat, calibrates rejection thresholds on validation
probes only, evaluates them unchanged on test probes, and saves the identity
manifest, best checkpoint, per-probe predictions, and a JSON summary.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from rfsensing.data.base import CSIDataModule
from rfsensing.eval.reid import (
    ReIDScores,
    calibrate_thresholds,
    detection_metrics,
    open_set_metrics,
    retrieval_metrics,
    score_gallery_probe,
)
from rfsensing.train.reid import ReIDModule


@dataclass
class ReIDResult:
    metrics: dict[str, float]
    thresholds: dict[str, float]
    checkpoint_path: Path
    log_dir: Path
    manifest_path: Path
    predictions_path: Path
    summary_path: Path


def _save_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    # allow_nan=False turns accidental NaN/Inf into a hard error instead of
    # silently writing invalid JSON.
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path


@torch.no_grad()
def _embed_loader(net: nn.Module, loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect raw ``net.embed`` features, restoring the prior train mode."""
    was_training = net.training
    net.eval()
    embeddings, labels = [], []
    for x, y in loader:
        embeddings.append(net.embed(x))
        labels.append(y)
    if was_training:
        net.train()
    embeddings = torch.cat(embeddings)
    if not torch.isfinite(embeddings).all():
        raise ValueError("encoder produced non-finite embeddings")
    return embeddings, torch.cat(labels)


def _predictions_rows(
    identity_names: list[str],
    scores: ReIDScores,
    probe_labels: torch.Tensor,
    known_mask: torch.Tensor,
    operating_points: dict[str, float],
) -> list[dict]:
    """One row per probe: identity, top match, rank, and thresholded calls."""
    rows = []
    enrolled = scores.identity_labels
    for i in range(probe_labels.numel()):
        label = int(probe_labels[i])
        ranked = scores.ranked_identities[i]
        rank_positions = (ranked == label).nonzero()
        rank = int(rank_positions[0]) + 1 if rank_positions.numel() else -1
        top_identity = identity_names[int(scores.top_identities[i])]
        top_score = float(scores.top_scores[i])
        row = {
            "probe_identity": identity_names[label],
            "known": bool(known_mask[i]),
            "top_identity": top_identity,
            "top_score": top_score,
            "rank": rank,
        }
        for point, threshold in operating_points.items():
            row[f"pred_{point}"] = (
                top_identity if top_score >= threshold else "unknown"
            )
        rows.append(row)
    assert enrolled.numel() > 0
    return rows


def _write_predictions(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _score_roles(
    net: nn.Module, loaders: dict
) -> tuple[ReIDScores, torch.Tensor, torch.Tensor]:
    gallery_z, gallery_y = _embed_loader(net, loaders["gallery"])
    known_z, known_y = _embed_loader(net, loaders["known_probes"])
    unknown_z, unknown_y = _embed_loader(net, loaders["unknown_probes"])
    scores = score_gallery_probe(
        gallery_z, gallery_y, torch.cat([known_z, unknown_z])
    )
    probe_labels = torch.cat([known_y, unknown_y])
    known_mask = torch.tensor(
        [True] * known_y.numel() + [False] * unknown_y.numel()
    )
    return scores, probe_labels, known_mask


def run_reid(
    net: nn.Module,
    dm: CSIDataModule,
    *,
    max_epochs: int = 50,
    name: str | None = None,
    seed: int = 42,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    triplet_margin: float = 0.3,
    triplet_weight: float = 1.0,
    far_target: float = 0.05,
    accelerator: str = "auto",
    runs_dir: str | Path = "runs",
) -> ReIDResult:
    """Train one Re-ID repeat and evaluate open-set metrics on the test roles."""
    manifest_seed = dm.split_manifest.seed
    if seed != manifest_seed:
        raise ValueError(
            f"seed {seed} does not match the DataModule split seed "
            f"{manifest_seed}; repeats must be internally consistent"
        )
    L.seed_everything(seed, workers=True)
    module = ReIDModule(
        net,
        num_train_identities=dm.output_dim,
        triplet_margin=triplet_margin,
        triplet_weight=triplet_weight,
        lr=lr,
        weight_decay=weight_decay,
    )
    experiment = name or type(net).__name__.lower()
    logger = TensorBoardLogger(str(runs_dir), name=f"{dm.name}/{experiment}")
    checkpoint = ModelCheckpoint(monitor="val/mAP", mode="max", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=dm)
    if checkpoint.best_model_score is None:
        raise RuntimeError("checkpoint monitor 'val/mAP' produced no score")
    state = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=True
    )["state_dict"]
    net.load_state_dict(
        {k.removeprefix("net."): v for k, v in state.items() if k.startswith("net.")}
    )
    net.cpu()

    log_dir = Path(trainer.log_dir)
    far_point = f"far{round(far_target * 100):02d}_threshold"

    # Thresholds come from validation scores only; test labels never enter.
    val_scores, _, val_known = _score_roles(net, dm.validation_loaders_by_role())
    thresholds = calibrate_thresholds(
        val_scores.top_scores, val_known, far_target=far_target
    )

    test_scores, test_labels, test_known = _score_roles(
        net, dm.test_loaders_by_role()
    )
    num_identities = test_scores.identity_labels.numel()
    capped = min(3, num_identities)
    retrieval = retrieval_metrics(
        test_scores,
        test_labels,
        test_known,
        ranks=(1, capped) if capped > 1 else (1,),
    )
    metrics = {
        "test/rank1": retrieval["rank1"],
        "test/rank3": retrieval[f"rank{capped}"],
        "test/mAP": retrieval["mAP"],
    }
    detection = detection_metrics(test_scores.top_scores, test_known)
    metrics["test/auroc"] = detection["auroc"]
    metrics["test/eer"] = detection["eer"]
    operating_points = {
        "eer_threshold": thresholds["eer_threshold"],
        far_point: thresholds["far_threshold"],
    }
    for point, threshold in operating_points.items():
        open_set = open_set_metrics(
            test_scores, test_labels, test_known, threshold
        )
        for key, value in open_set.items():
            metrics[f"test/{point}/{key}"] = value

    manifest_path = _save_json(
        log_dir / "manifest.json",
        {"dataset": dm.name, **dm.split_manifest.to_dict()},
    )
    _save_json(
        log_dir / "config.json",
        {
            "net": type(net).__name__,
            "max_epochs": max_epochs,
            "seed": seed,
            "lr": lr,
            "weight_decay": weight_decay,
            "triplet_margin": triplet_margin,
            "triplet_weight": triplet_weight,
            "far_target": far_target,
        },
    )
    predictions_path = _write_predictions(
        log_dir / "predictions.csv",
        _predictions_rows(
            dm.identity_names, test_scores, test_labels, test_known,
            operating_points,
        ),
    )
    summary_path = _save_json(
        log_dir / "summary.json",
        {
            "metrics": metrics,
            "thresholds": {**thresholds, "calibrated_on": "validation"},
            "best_val_mAP": float(checkpoint.best_model_score),
            "checkpoint": str(checkpoint.best_model_path),
            "manifest": str(manifest_path),
            "predictions": str(predictions_path),
        },
    )
    return ReIDResult(
        metrics=metrics,
        thresholds=thresholds,
        checkpoint_path=Path(checkpoint.best_model_path),
        log_dir=log_dir,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
    )
