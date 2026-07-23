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
