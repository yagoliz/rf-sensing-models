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
    assert 'detection_score="top_gap"' in source
