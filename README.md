# rfsensing

RF-sensing (WiFi CSI) models and benchmarks — a clean reimplementation of
[SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) as a
PyTorch/Lightning library, for the RF Sensing for Individual and Group
Classification project.

## Setup

```bash
uv sync
```

Datasets are expected at `../../data/` relative to this repo (the project
data root containing `UT_HAR/`, `NTU-Fi_HAR/`, `NTU-Fi-HumanID/`,
`Widardata/`). Override with the `RFSENSING_DATA` env var for tests, or
pass `root=` explicitly.

## Usage

```python
from rfsensing import data, models, train

dm = data.build("ntu_fi_har", root=DATA_DIR, batch_size=64)
net = models.build("vit", in_shape=dm.sample_shape, num_classes=dm.num_classes)
result = train.run(net, dm, max_epochs=50, name="vit-ntu-har")
result.metrics  # {'test/acc': ..., 'test/loss': ...}
```

- Datasets: `ut_har`, `ntu_fi_har`, `ntu_fi_humanid`, `widar`, `synthetic`
- Models: `mlp`, `lenet`, `lstm` (use `bidirectional=True` for BiLSTM),
  `resnet18`, `vit`
- Register your own: `@models.register("myname")` on a class taking
  `(in_shape, num_classes, **kwargs)`; likewise `@data.register` for
  DataModules implementing the `CSIDataModule` contract
  (`sample_shape`, `num_classes`, `class_names`).

TensorBoard logs land in `runs/`: `uv run tensorboard --logdir runs`.

## Notebooks

`notebooks/00_data_exploration` explores the datasets;
`01`–`04` run the model zoo per dataset and tabulate results.
Notebooks are paired `.py` (jupytext percent) + `.ipynb`. After editing a
`.py`, regenerate with `uv run jupytext --to ipynb notebooks/<name>.py`.

## Tests

```bash
uv run pytest              # includes real-data tests when data/ exists
uv run pytest -m "not data"  # library-only, no datasets needed
```