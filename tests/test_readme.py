from pathlib import Path


def test_readme_describes_the_broader_research_platform():
    path = Path(__file__).resolve().parents[1] / "README.md"
    source = path.read_text()
    introduction = source.split("## What works today", maxsplit=1)[0]

    assert "modelling and evaluation layer" in introduction
    assert "SenseFi" not in introduction
    assert "## What works today" in source
    assert "## Research direction" in source
    assert "## Origins and compatibility" in source
    assert source.index("## Research direction") < source.index(
        "## Origins and compatibility"
    )
    assert 'data.build("wimans", target="classification"' in source
    assert 'data.build("wimans", target="regression"' in source
