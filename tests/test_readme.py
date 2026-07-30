import re
from pathlib import Path

import torch

from rfsensing import data


def test_readme_describes_the_broader_research_platform(tmp_path):
    path = Path(__file__).resolve().parents[1] / "README.md"
    source = path.read_text()
    introduction = source.split("## What works today", maxsplit=1)[0]
    current = source.split("## What works today", maxsplit=1)[1].split(
        "## Installation", maxsplit=1
    )[0]
    roadmap = source.split("## Research direction", maxsplit=1)[1].split(
        "## Extending `rfsensing`", maxsplit=1
    )[0]

    assert "modelling and evaluation layer" in introduction
    assert "SenseFi" not in introduction
    assert "](../../MAIN_PROJECT.md)" not in introduction
    assert "## What works today" in source
    assert "## Research direction" in source
    assert "## Origins and compatibility" in source
    assert source.index("## Research direction") < source.index(
        "## Origins and compatibility"
    )
    assert 'data.build("wimans", target="classification"' in source
    assert 'data.build("wimans", target="regression"' in source
    assert "BVP" in current
    assert "re-identification" in current
    assert "WhoFi" in roadmap
    for option in (
        '"pad_side"',
        '"environments"',
        '"wifi_bands"',
        'split_strategy="random"',
    ):
        assert option in source

    for block in re.findall(r"```python\n(.*?)```", source, flags=re.DOTALL):
        compile(block, str(path), "exec")

    data_module_section = source.split(
        "Register a DataModule with:", maxsplit=1
    )[1]
    data_module_block = data_module_section.split(
        "```python\n", maxsplit=1
    )[1].split("```", maxsplit=1)[0]
    exec(data_module_block, {})
    dm = data.build("my_dataset", root=tmp_path, batch_size=4)
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, 3, 30, 100)
    assert y.dtype == torch.int64


def test_readme_documents_open_set_reid():
    path = Path(__file__).resolve().parents[1] / "README.md"
    source = path.read_text()
    roadmap = source.split("## Research direction", maxsplit=1)[1].split(
        "## Extending `rfsensing`", maxsplit=1
    )[0]

    assert "ntu_fi_humanid_reid" in source
    assert "7/2/1/3/1" in source
    assert "P×K" in source
    assert "batch-hard triplet" in source
    assert "cross-entropy" in source
    for metric in ("rank-1", "rank-3", "mAP", "AUROC", "EER", "DIR", "FAR"):
        assert metric in source
    assert "rank-5" in source  # explains why rank-3 replaces it
    assert "calibrated" in source and "validation" in source
    assert "run_reid_repeats" in source
    assert '"resnet18"' in source
    assert '"vit"' in source
    for artifact in (
        "manifest.json",
        "predictions.csv",
        "summary.json",
        "aggregate_summary.json",
    ):
        assert artifact in source
    assert "06_open_set_person_reid" in source
    assert "WhoFi" in source
    assert "future work" in source
    # WhoFi reproduction and richer objectives stay on the roadmap.
    for item in ("WhoFi", "ArcFace", "supervised contrastive", "HDF5"):
        assert item in roadmap

    assert "supcon" in source
    assert "top_gap" in source

    from rfsensing import train

    for api in (
        "run_reid",
        "run_reid_repeats",
        "ReIDResult",
        "RepeatedReIDResult",
        "ReIDModule",
        "batch_hard_triplet_loss",
        "supcon_loss",
    ):
        assert hasattr(train, api)
