from pathlib import Path


def test_wimans_counting_notebook_source_contract():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "05_wimans_counting.py"
    assert path.is_file()
    source = path.read_text()
    compile(source, str(path), "exec")
    assert 'target="classification"' in source
    assert 'target="regression"' in source
    assert '"split_strategy": "group"' in source
    assert "test/within_1" in source
    assert "best_score" in source
    assert '["test/mae"].idxmin()' not in source


def test_open_set_reid_notebook_source_contract():
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "06_open_set_person_reid.py"
    )
    assert path.is_file()
    source = path.read_text()
    compile(source, str(path), "exec")
    assert "ntu_fi_humanid_reid" in source
    assert '"resnet18"' in source
    assert '"vit"' in source
    assert "SEEDS = (42, 43, 44)" in source
    assert "run_reid_repeats" in source
    for metric in ("rank1", "rank3", "mAP", "auroc", "eer"):
        assert metric in source
    assert "test/eer_threshold/dir" in source
    assert "test/far05_threshold/far" in source
    assert "RUN_TRAINING" in source
    assert "top_score" in source  # known/unknown score distribution plot
    assert "smoke test" in source
    assert "WhoFi" in source
    assert 'objective="supcon"' in source
    # The variant cell reproduces the committed aggregates (top score);
    # the superseded top-gap experiment is documented in the markdown.
    assert 'detection_score="top_score"' in source
    assert "top_gap" in source
    # Embedding-space comparison: 2-D projection + cosine histograms.
    assert "project(" in source
    assert "tsne" in source
    assert "intra" in source and "inter" in source


def test_reid_variant_analysis_notebook_source_contract():
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "07_reid_variant_analysis.py"
    )
    assert path.is_file()
    source = path.read_text()
    compile(source, str(path), "exec")
    # Reads notebook 06's artifacts; never retrains them.
    assert "aggregate_summary.json" in source
    assert "extended-reid" in source
    assert "supcon-gap-reid" in source
    # Detection-score forensics compare both scorers per probe.
    assert "auroc_gap" in source and "auroc_top" in source
    assert "roc_auc_score" in source
    # ArcFace benchmark: opt-in training, paired with the top score.
    assert "RUN_ARCFACE" in source
    assert 'objective="arcface"' in source
    assert 'detection_score="top_score"' in source
    assert "arcface-reid" in source
    # Per-seed paired analysis and cosine-geometry comparison.
    assert "per_seed_table" in source
    assert "intra" in source and "inter" in source
