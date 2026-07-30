import math

import pytest
import torch

from rfsensing.eval.reid import (
    ReIDScores,
    calibrate_thresholds,
    detection_metrics,
    open_set_metrics,
    retrieval_metrics,
    score_gallery_probe,
)


def _scores(gallery, gallery_labels, probes):
    return score_gallery_probe(
        torch.tensor(gallery, dtype=torch.float32),
        torch.tensor(gallery_labels),
        torch.tensor(probes, dtype=torch.float32),
    )


# --- score_gallery_probe ---


def test_scoring_l2_normalizes_embeddings():
    scaled = _scores([[2.0, 0.0]], [0], [[5.0, 0.0]])
    unit = _scores([[1.0, 0.0]], [0], [[1.0, 0.0]])
    assert torch.allclose(scaled.sample_scores, unit.sample_scores)
    assert scaled.sample_scores[0, 0].item() == pytest.approx(1.0)


def test_scoring_cosine_sample_scores():
    scores = _scores(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 1, 1], [[1.0, 0.0]]
    )
    expected = torch.tensor([[1.0, 0.0, math.sqrt(0.5)]])
    assert torch.allclose(scores.sample_scores, expected, atol=1e-6)


def test_scoring_identity_scores_use_max_aggregation():
    scores = _scores(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 1, 1], [[1.0, 0.0]]
    )
    assert torch.equal(scores.identity_labels, torch.tensor([0, 1]))
    expected = torch.tensor([[1.0, math.sqrt(0.5)]])
    assert torch.allclose(scores.identity_scores, expected, atol=1e-6)
    assert scores.top_identities[0].item() == 0
    assert scores.top_scores[0].item() == pytest.approx(1.0)


def test_scoring_breaks_ties_by_ascending_identity():
    scores = _scores([[1.0, 0.0], [1.0, 0.0]], [1, 0], [[1.0, 0.0]])
    assert scores.ranked_identities[0].tolist() == [0, 1]
    assert scores.top_identities[0].item() == 0


def test_scoring_rejects_bad_inputs():
    good = torch.eye(2)
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="2-D"):
        score_gallery_probe(good.flatten(), labels, good)
    with pytest.raises(ValueError, match="finite"):
        score_gallery_probe(good * float("nan"), labels, good)
    with pytest.raises(ValueError, match="finite"):
        score_gallery_probe(good, labels, good * float("inf"))
    with pytest.raises(ValueError, match="empty"):
        score_gallery_probe(torch.empty(0, 2), torch.empty(0), good)
    with pytest.raises(ValueError, match="empty"):
        score_gallery_probe(good, labels, torch.empty(0, 2))
    with pytest.raises(ValueError, match="labels"):
        score_gallery_probe(good, torch.tensor([0]), good)


# --- retrieval_metrics ---


def _retrieval_fixture():
    # Probe 0 (label 0): gallery ranking [id0 (rel), id1, id0 (rel)]
    # by angles 0, 30, 60 degrees.
    gallery = [
        [1.0, 0.0],
        [math.cos(math.radians(30)), math.sin(math.radians(30))],
        [math.cos(math.radians(60)), math.sin(math.radians(60))],
    ]
    gallery_labels = [0, 1, 0]
    probes = [[1.0, 0.0], [0.0, 1.0]]
    probe_labels = torch.tensor([0, 1])
    known = torch.tensor([True, True])
    return _scores(gallery, gallery_labels, probes), probe_labels, known


def test_retrieval_rank_accuracy():
    scores, probe_labels, known = _retrieval_fixture()
    metrics = retrieval_metrics(scores, probe_labels, known, ranks=(1, 2))
    # Probe 0 top identity is 0 (correct); probe 1 (label 1) top score:
    # id1 at 60 deg from probe -> cos(60) = 0.5; id0 best cos(30) = 0.866,
    # so probe 1 rank-1 is wrong but rank-2 correct.
    assert metrics["rank1"] == pytest.approx(0.5)
    assert metrics["rank2"] == pytest.approx(1.0)


def test_retrieval_map_sample_level():
    scores, probe_labels, known = _retrieval_fixture()
    metrics = retrieval_metrics(scores, probe_labels, known, ranks=(1,))
    # Probe 0 ranked gallery samples: [rel, non, rel] -> AP = (1 + 2/3)/2.
    # Probe 1 ranked samples: [non, rel, non] -> AP = 1/2.
    expected = ((1.0 + 2.0 / 3.0) / 2.0 + 1.0 / 2.0) / 2.0
    assert metrics["mAP"] == pytest.approx(expected)


def test_retrieval_map_uses_known_probes_only():
    gallery = [
        [1.0, 0.0],
        [math.cos(math.radians(30)), math.sin(math.radians(30))],
        [math.cos(math.radians(60)), math.sin(math.radians(60))],
    ]
    gallery_labels = [0, 1, 0]
    known_probes = [[1.0, 0.0], [0.0, 1.0]]
    unknown_probe = [[-1.0, 0.0]]
    with_unknown = retrieval_metrics(
        _scores(gallery, gallery_labels, known_probes + unknown_probe),
        torch.tensor([0, 1, 5]),
        torch.tensor([True, True, False]),
        ranks=(1,),
    )
    known_only = retrieval_metrics(
        _scores(gallery, gallery_labels, known_probes),
        torch.tensor([0, 1]),
        torch.tensor([True, True]),
        ranks=(1,),
    )
    assert with_unknown["mAP"] == pytest.approx(known_only["mAP"])
    assert with_unknown["rank1"] == pytest.approx(known_only["rank1"])


def test_retrieval_rejects_invalid_ranks():
    scores, probe_labels, known = _retrieval_fixture()
    with pytest.raises(ValueError, match="rank"):
        retrieval_metrics(scores, probe_labels, known, ranks=(0,))
    with pytest.raises(ValueError, match="rank"):
        retrieval_metrics(scores, probe_labels, known, ranks=(3,))


def test_retrieval_rejects_probe_gallery_mismatch():
    scores, probe_labels, known = _retrieval_fixture()
    with pytest.raises(ValueError, match="gallery"):
        retrieval_metrics(scores, torch.tensor([0, 5]), known, ranks=(1,))
    with pytest.raises(ValueError, match="known probes"):
        retrieval_metrics(
            scores, torch.tensor([0, 0]), known, ranks=(1,)
        )


def test_retrieval_requires_known_probes():
    scores, probe_labels, _ = _retrieval_fixture()
    with pytest.raises(ValueError, match="known"):
        retrieval_metrics(
            scores, probe_labels, torch.tensor([False, False]), ranks=(1,)
        )


# --- detection_metrics ---


def test_detection_separable_scores():
    top_scores = torch.tensor([0.8, 0.9, 0.1, 0.2])
    known = torch.tensor([True, True, False, False])
    metrics = detection_metrics(top_scores, known)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["eer"] == pytest.approx(0.0)


def test_detection_overlapping_scores():
    top_scores = torch.tensor([0.9, 0.4, 0.6])
    known = torch.tensor([True, True, False])
    metrics = detection_metrics(top_scores, known)
    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["eer"] == pytest.approx(0.25)


def test_detection_requires_both_classes():
    with pytest.raises(ValueError, match="unknown"):
        detection_metrics(torch.tensor([0.5, 0.6]), torch.tensor([True, True]))
    with pytest.raises(ValueError, match="known"):
        detection_metrics(
            torch.tensor([0.5, 0.6]), torch.tensor([False, False])
        )


# --- calibrate_thresholds ---


def test_calibration_eer_threshold():
    known_scores = [0.15, 0.3, 0.4]
    unknown_scores = [0.1, 0.2]
    top_scores = torch.tensor(known_scores + unknown_scores)
    known = torch.tensor([True] * 3 + [False] * 2)
    thresholds = calibrate_thresholds(top_scores, known)
    assert thresholds["eer_threshold"] == pytest.approx(0.2)
    assert thresholds["eer"] == pytest.approx((0.5 + 1.0 / 3.0) / 2.0)


def test_calibration_far_threshold_below_target_when_exact_impossible():
    top_scores = torch.tensor([0.5, 0.7, 0.9, 0.2, 0.6])
    known = torch.tensor([True, True, True, False, False])
    thresholds = calibrate_thresholds(top_scores, known, far_target=0.05)
    # No threshold reaches exactly 5% FAR with 2 unknowns; the calibrated
    # threshold must reject both unknowns and keep the most knowns.
    assert thresholds["far_threshold"] == pytest.approx(0.7)
    assert thresholds["far_achieved"] == pytest.approx(0.0)
    assert thresholds["far_achieved"] < 0.05
    assert thresholds["far_target"] == pytest.approx(0.05)


def test_calibration_far_tie_selects_higher_threshold():
    known_scores = [0.8, 0.9]
    unknown_scores = [0.1] * 19 + [0.45]
    top_scores = torch.tensor(known_scores + unknown_scores)
    known = torch.tensor([True] * 2 + [False] * 20)
    thresholds = calibrate_thresholds(top_scores, known, far_target=0.05)
    # 0.45 and 0.8 both accept every known; 0.45 keeps FAR at exactly the
    # 5% target, but the tie must resolve to the stricter 0.8.
    assert thresholds["far_threshold"] == pytest.approx(0.8)
    assert thresholds["far_achieved"] == pytest.approx(0.0)


def test_calibration_can_reject_everything():
    top_scores = torch.tensor([0.5, 0.6, 0.7, 0.8])
    known = torch.tensor([True, False, False, False])
    thresholds = calibrate_thresholds(top_scores, known, far_target=0.05)
    # Every unknown outscores the known probe: only reject-all is feasible.
    assert thresholds["far_threshold"] > 0.8
    assert thresholds["far_achieved"] == pytest.approx(0.0)


def test_calibration_rejects_invalid_inputs():
    top_scores = torch.tensor([0.5, 0.6])
    known = torch.tensor([True, False])
    with pytest.raises(ValueError, match="far_target"):
        calibrate_thresholds(top_scores, known, far_target=0.0)
    with pytest.raises(ValueError, match="far_target"):
        calibrate_thresholds(top_scores, known, far_target=1.0)
    with pytest.raises(ValueError, match="finite"):
        calibrate_thresholds(torch.tensor([0.5, float("nan")]), known)
    with pytest.raises(ValueError, match="unknown"):
        calibrate_thresholds(top_scores, torch.tensor([True, True]))


# --- open_set_metrics ---


def _open_set_fixture():
    gallery = [[1.0, 0.0], [0.0, 1.0]]
    gallery_labels = [0, 1]
    # p0: label 0, correct top (score 1). p1: label 0, top is id 1 (0.8).
    # p2: label 1, correct top (score 1). p3: unknown, top score ~0.707.
    probes = [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [0.707, 0.707]]
    scores = _scores(gallery, gallery_labels, probes)
    probe_labels = torch.tensor([0, 0, 1, 5])
    known = torch.tensor([True, True, True, False])
    return scores, probe_labels, known


def test_open_set_metrics_at_threshold():
    scores, probe_labels, known = _open_set_fixture()
    metrics = open_set_metrics(scores, probe_labels, known, threshold=0.75)
    assert metrics["far"] == pytest.approx(0.0)
    assert metrics["unknown_rejection"] == pytest.approx(1.0)
    assert metrics["known_acceptance"] == pytest.approx(1.0)
    assert metrics["dir"] == pytest.approx(2.0 / 3.0)
    assert metrics["accepted_top1"] == pytest.approx(2.0 / 3.0)


def test_open_set_metrics_accepting_unknown():
    scores, probe_labels, known = _open_set_fixture()
    metrics = open_set_metrics(scores, probe_labels, known, threshold=0.5)
    assert metrics["far"] == pytest.approx(1.0)
    assert metrics["unknown_rejection"] == pytest.approx(0.0)


def test_open_set_all_rejected_gives_zero_dir():
    scores, probe_labels, known = _open_set_fixture()
    metrics = open_set_metrics(scores, probe_labels, known, threshold=2.0)
    assert metrics["dir"] == pytest.approx(0.0)
    assert metrics["known_acceptance"] == pytest.approx(0.0)
    assert metrics["accepted_top1"] == pytest.approx(0.0)
    assert metrics["far"] == pytest.approx(0.0)
    assert metrics["unknown_rejection"] == pytest.approx(1.0)


def test_open_set_requires_known_and_unknown_probes():
    scores, probe_labels, known = _open_set_fixture()
    with pytest.raises(ValueError, match="unknown"):
        open_set_metrics(
            scores, probe_labels, torch.ones(4, dtype=torch.bool), 0.5
        )


def test_reid_scores_is_frozen():
    scores, _, _ = _open_set_fixture()
    assert isinstance(scores, ReIDScores)
    with pytest.raises(AttributeError):
        scores.top_scores = torch.zeros(1)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
def test_eval_pipeline_accepts_non_cpu_embeddings():
    generator = torch.Generator().manual_seed(0)
    gallery = torch.randn(8, 4, generator=generator).to("mps")
    gallery_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    probes = torch.randn(10, 4, generator=generator).to("mps")
    probe_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 9, 9])
    known = torch.tensor([True] * 8 + [False] * 2)
    scores = score_gallery_probe(gallery, gallery_labels, probes)
    retrieval = retrieval_metrics(scores, probe_labels, known, ranks=(1, 3))
    assert 0.0 <= retrieval["mAP"] <= 1.0
    detection = detection_metrics(scores.top_scores, known)
    assert 0.0 <= detection["auroc"] <= 1.0
    thresholds = calibrate_thresholds(scores.top_scores, known)
    open_set = open_set_metrics(
        scores, probe_labels, known, thresholds["eer_threshold"]
    )
    assert 0.0 <= open_set["dir"] <= 1.0


# --- top-gap detection scores ---


def test_scoring_top_gaps():
    scores = _scores(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 1, 1], [[1.0, 0.0]]
    )
    # identity scores [1.0, sqrt(0.5)] -> gap = 1 - sqrt(0.5)
    assert scores.top_gaps[0].item() == pytest.approx(
        1.0 - math.sqrt(0.5), abs=1e-6
    )


def test_scoring_top_gaps_single_identity_falls_back_to_top_score():
    scores = _scores([[1.0, 0.0]], [0], [[0.6, 0.8]])
    assert torch.allclose(scores.top_gaps, scores.top_scores)


def test_open_set_metrics_with_gap_detection_scores():
    scores, probe_labels, known = _open_set_fixture()
    # Unknown probe [0.707, 0.707] ties both identities: top score ~0.707
    # but gap ~0. A top-score threshold of 0.7 accepts it; a gap threshold
    # of 0.15 rejects it while keeping all known probes.
    top_based = open_set_metrics(scores, probe_labels, known, threshold=0.7)
    assert top_based["far"] == pytest.approx(1.0)
    gap_based = open_set_metrics(
        scores,
        probe_labels,
        known,
        threshold=0.15,
        detection_scores=scores.top_gaps,
    )
    assert gap_based["far"] == pytest.approx(0.0)
    assert gap_based["known_acceptance"] == pytest.approx(1.0)
    assert gap_based["dir"] == pytest.approx(2.0 / 3.0)


def test_open_set_metrics_rejects_bad_detection_scores():
    scores, probe_labels, known = _open_set_fixture()
    with pytest.raises(ValueError, match="detection"):
        open_set_metrics(
            scores,
            probe_labels,
            known,
            threshold=0.5,
            detection_scores=torch.tensor([0.5, 0.6]),
        )
