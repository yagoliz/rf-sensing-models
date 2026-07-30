"""Gallery-probe retrieval and open-set rejection metrics.

Independent of any dataset: callers pass embeddings, stable identity labels,
and known/unknown probe masks. Probes are scored against every gallery sample
with cosine similarity; per-identity scores aggregate gallery samples by
maximum. A probe is accepted as its top-ranked identity when its top score
meets the operating threshold, otherwise predicted unknown.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReIDScores:
    gallery_labels: torch.Tensor
    identity_labels: torch.Tensor
    sample_scores: torch.Tensor       # (num_probes, num_gallery_samples)
    identity_scores: torch.Tensor     # (num_probes, num_enrolled_identities)
    ranked_identities: torch.Tensor   # descending score, ties -> ascending id
    top_scores: torch.Tensor
    top_identities: torch.Tensor
    top_gaps: torch.Tensor            # top-1 minus top-2 identity score


def _check_embeddings(embeddings: torch.Tensor, what: str) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"{what} embeddings must be 2-D, got {embeddings.ndim}-D")
    if embeddings.shape[0] == 0:
        raise ValueError(f"{what} embeddings are empty")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{what} embeddings must be finite")
    return embeddings.float()


def _check_scores(scores: torch.Tensor, what: str) -> torch.Tensor:
    scores = torch.as_tensor(scores).float().reshape(-1)
    if scores.numel() == 0:
        raise ValueError(f"{what} must not be empty")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{what} must be finite")
    return scores


def _check_known_mask(known_mask: torch.Tensor, count: int) -> torch.Tensor:
    known_mask = torch.as_tensor(known_mask).bool().reshape(-1)
    if known_mask.numel() != count:
        raise ValueError(
            f"known mask has {known_mask.numel()} entries for {count} probes"
        )
    if not known_mask.any():
        raise ValueError("no known probes present")
    if known_mask.all():
        raise ValueError("no unknown probes present")
    return known_mask


def score_gallery_probe(
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    probe_embeddings: torch.Tensor,
) -> ReIDScores:
    """Cosine-score every probe against the gallery, aggregated per identity."""
    gallery = _check_embeddings(gallery_embeddings, "gallery")
    probes = _check_embeddings(probe_embeddings, "probe")
    gallery_labels = (
        torch.as_tensor(gallery_labels).long().reshape(-1).to(gallery.device)
    )
    if gallery_labels.numel() != gallery.shape[0]:
        raise ValueError(
            f"gallery labels count {gallery_labels.numel()} does not match "
            f"{gallery.shape[0]} gallery embeddings"
        )
    if gallery.shape[1] != probes.shape[1]:
        raise ValueError(
            f"embedding dims differ: gallery {gallery.shape[1]}, "
            f"probe {probes.shape[1]}"
        )
    gallery = torch.nn.functional.normalize(gallery, dim=1)
    probes = torch.nn.functional.normalize(probes, dim=1)
    sample_scores = probes @ gallery.T
    identity_labels = torch.unique(gallery_labels)  # sorted ascending
    identity_scores = torch.stack(
        [
            sample_scores[:, gallery_labels == identity].amax(dim=1)
            for identity in identity_labels
        ],
        dim=1,
    )
    # Stable descending sort over ascending-identity columns: ties resolve to
    # the lowest stable identity ID.
    order = torch.argsort(identity_scores, dim=1, descending=True, stable=True)
    ranked_identities = identity_labels[order]
    top_scores = identity_scores.amax(dim=1)
    if identity_labels.numel() > 1:
        sorted_scores = identity_scores.sort(dim=1, descending=True).values
        # A relative detection score: large when one enrolled identity
        # clearly wins, near zero when the probe is equidistant from several.
        top_gaps = sorted_scores[:, 0] - sorted_scores[:, 1]
    else:
        top_gaps = top_scores.clone()  # no runner-up exists
    return ReIDScores(
        gallery_labels=gallery_labels,
        identity_labels=identity_labels,
        sample_scores=sample_scores,
        identity_scores=identity_scores,
        ranked_identities=ranked_identities,
        top_scores=top_scores,
        top_identities=ranked_identities[:, 0],
        top_gaps=top_gaps,
    )


def _known_probe_labels(
    scores: ReIDScores,
    probe_labels: torch.Tensor,
    known_mask: torch.Tensor,
    *,
    require_unknown: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = scores.sample_scores.device
    probe_labels = torch.as_tensor(probe_labels).long().reshape(-1).to(device)
    if probe_labels.numel() != scores.sample_scores.shape[0]:
        raise ValueError(
            f"probe labels count {probe_labels.numel()} does not match "
            f"{scores.sample_scores.shape[0]} probes"
        )
    known_mask = torch.as_tensor(known_mask).bool().reshape(-1).to(device)
    if known_mask.numel() != probe_labels.numel():
        raise ValueError("known mask length must match probe labels")
    if not known_mask.any():
        raise ValueError("no known probes present")
    if require_unknown and known_mask.all():
        raise ValueError("no unknown probes present")
    enrolled = set(scores.identity_labels.tolist())
    known_labels = set(probe_labels[known_mask].tolist())
    if not known_labels <= enrolled:
        raise ValueError(
            f"known probe labels {sorted(known_labels - enrolled)} are "
            "absent from the gallery"
        )
    if not enrolled <= known_labels:
        raise ValueError(
            f"gallery identities {sorted(enrolled - known_labels)} have no "
            "known probes"
        )
    return probe_labels, known_mask


def retrieval_metrics(
    scores: ReIDScores,
    probe_labels: torch.Tensor,
    known_mask: torch.Tensor,
    ranks: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """Rank-k accuracy and sample-level mAP over known probes only."""
    probe_labels, known_mask = _known_probe_labels(
        scores, probe_labels, known_mask
    )
    num_identities = scores.identity_labels.numel()
    for k in ranks:
        if not 1 <= k <= num_identities:
            raise ValueError(
                f"rank {k} is invalid for {num_identities} enrolled identities"
            )
    ranked = scores.ranked_identities[known_mask]
    labels = probe_labels[known_mask]
    metrics: dict[str, float] = {}
    for k in ranks:
        hits = (ranked[:, :k] == labels.unsqueeze(1)).any(dim=1)
        metrics[f"rank{k}"] = hits.float().mean().item()

    sample_scores = scores.sample_scores[known_mask]
    # Stable descending sort keeps ascending gallery-sample order on ties.
    order = torch.argsort(sample_scores, dim=1, descending=True, stable=True)
    relevant = (
        scores.gallery_labels[order] == labels.unsqueeze(1)
    ).float()
    positions = torch.arange(
        1, relevant.shape[1] + 1, dtype=torch.float32, device=relevant.device
    ).unsqueeze(0)
    precision = relevant.cumsum(dim=1) / positions
    average_precision = (precision * relevant).sum(dim=1) / relevant.sum(dim=1)
    metrics["mAP"] = average_precision.mean().item()
    return metrics


def _roc(top_scores: torch.Tensor, known_mask: torch.Tensor):
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(
        known_mask.cpu().numpy(),
        top_scores.cpu().numpy(),
        drop_intermediate=False,
    )
    return fpr, tpr, thresholds


def detection_metrics(
    top_scores: torch.Tensor, known_mask: torch.Tensor
) -> dict[str, float]:
    """AUROC and EER for known-versus-unknown detection scores."""
    from sklearn.metrics import roc_auc_score

    top_scores = _check_scores(top_scores, "top scores")
    known_mask = _check_known_mask(known_mask, top_scores.numel()).to(
        top_scores.device
    )
    fpr, tpr, _ = _roc(top_scores, known_mask)
    fnr = 1.0 - tpr
    eer_index = int(abs(fpr - fnr).argmin())  # first minimum: deterministic
    return {
        "auroc": float(
            roc_auc_score(known_mask.cpu().numpy(), top_scores.cpu().numpy())
        ),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
    }


def calibrate_thresholds(
    top_scores: torch.Tensor,
    known_mask: torch.Tensor,
    far_target: float = 0.05,
) -> dict[str, float]:
    """Select EER and FAR-constrained accept thresholds from validation scores.

    A probe is accepted when ``score >= threshold``. The FAR-constrained
    threshold maximizes known acceptance subject to empirical FAR <= target;
    ties select the higher (stricter) threshold. If the target is unreachable
    exactly, the achieved FAR lies below it.
    """
    if not 0.0 < far_target < 1.0:
        raise ValueError(f"far_target must be in (0, 1), got {far_target}")
    top_scores = _check_scores(top_scores, "top scores")
    known_mask = _check_known_mask(known_mask, top_scores.numel()).to(
        top_scores.device
    )
    fpr, tpr, thresholds = _roc(top_scores, known_mask)
    fnr = 1.0 - tpr
    eer_index = int(abs(fpr - fnr).argmin())
    eer_threshold = float(thresholds[eer_index])
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)

    known_scores = top_scores[known_mask]
    unknown_scores = top_scores[~known_mask]
    reject_all = float(top_scores.max()) + 1.0
    candidates = sorted(
        set(top_scores.tolist()) | {reject_all}
    )
    best: tuple[float, float, float] | None = None  # (acceptance, threshold, far)
    for threshold in candidates:
        far = (unknown_scores >= threshold).float().mean().item()
        if far > far_target:
            continue
        acceptance = (known_scores >= threshold).float().mean().item()
        if (
            best is None
            or acceptance > best[0]
            or (acceptance == best[0] and threshold > best[1])
        ):
            best = (acceptance, threshold, far)
    assert best is not None  # reject_all always satisfies the constraint
    return {
        "eer_threshold": eer_threshold,
        "eer": eer,
        "far_threshold": best[1],
        "far_achieved": best[2],
        "far_target": far_target,
    }


def open_set_metrics(
    scores: ReIDScores,
    probe_labels: torch.Tensor,
    known_mask: torch.Tensor,
    threshold: float,
    *,
    detection_scores: torch.Tensor | None = None,
) -> dict[str, float]:
    """Acceptance/rejection metrics at a fixed operating threshold.

    DIR counts a known probe only when it is accepted *and* its top-ranked
    identity is correct, so rejecting everything cannot look successful.
    ``detection_scores`` overrides the acceptance score per probe (default:
    ``scores.top_scores``); e.g. pass ``scores.top_gaps`` for gap-based
    rejection. The predicted identity is always the top-ranked one.
    """
    probe_labels, known_mask = _known_probe_labels(
        scores, probe_labels, known_mask, require_unknown=True
    )
    if detection_scores is None:
        detection_scores = scores.top_scores
    else:
        detection_scores = _check_scores(detection_scores, "detection scores")
        if detection_scores.numel() != probe_labels.numel():
            raise ValueError(
                f"detection scores count {detection_scores.numel()} does not "
                f"match {probe_labels.numel()} probes"
            )
    accepted = detection_scores.to(probe_labels.device) >= threshold
    correct = scores.top_identities == probe_labels
    known_accepted = accepted[known_mask]
    far = accepted[~known_mask].float().mean().item()
    num_accepted_known = int(known_accepted.sum())
    accepted_top1 = (
        (correct[known_mask] & known_accepted).float().sum().item()
        / num_accepted_known
        if num_accepted_known
        else 0.0
    )
    return {
        "dir": (known_accepted & correct[known_mask]).float().mean().item(),
        "far": far,
        "unknown_rejection": 1.0 - far,
        "known_acceptance": known_accepted.float().mean().item(),
        "accepted_top1": accepted_top1,
    }
