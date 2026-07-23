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

- Datasets: `ut_har`, `ntu_fi_har`, `ntu_fi_humanid`, `widar`, `wimans`,
  `synthetic`
- Models: `mlp`, `lenet`, `lstm` (use `bidirectional=True` for BiLSTM),
  `resnet18`, `vit`
- Register your own: `@models.register("myname")` on a class taking
  `(in_shape, num_classes, **kwargs)`; likewise `@data.register` for
  DataModules implementing the `CSIDataModule` contract
  (`sample_shape`, `num_classes`, `class_names`, `output_dim`).

## WiMANS people counting

The WiMANS DataModule supports both ordered classification of counts 0–5 and
scalar count regression. It requires the complete amplitude dataset and accepts
either of these layouts:

```text
../../data/WiMANS/annotation.csv
../../data/WiMANS/wifi_csi/amp/*.npy  # official layout
../../data/WiMANS/amp/*.npy           # flat layout
```

Group-held-out splitting is the default: every `act_<group>_*` user
combination remains in exactly one of train, validation, or test. A random,
count-stratified split is available with `split_strategy="random"` when direct
comparison with less restrictive protocols is needed.

```python
common = {
    "root": DATA_DIR,
    "time_steps": 300,
    "pooling": "mean",          # "mean" or "max"
    "normalization": "train",   # "train", "sample", or "none"
    "split_strategy": "group",
    "split_seed": 42,
}

dm = data.build("wimans", target="classification", **common)
net = models.build("lstm", in_shape=dm.sample_shape, num_classes=dm.output_dim)
classification = train.run(net, dm, max_epochs=30, name="wimans-classification")

dm = data.build("wimans", target="regression", **common)
net = models.build("lstm", in_shape=dm.sample_shape, num_classes=dm.output_dim)
regression = train.run(net, dm, max_epochs=30, name="wimans-regression")
```

Classification reports exact accuracy, MAE, and ±1-person accuracy. Regression
reports raw MAE, ±1-person accuracy, and rounded exact accuracy; it also
automatically checkpoints on minimum validation MAE.

TensorBoard logs land in `runs/`: `uv run tensorboard --logdir runs`.

## Notebooks

`notebooks/00_data_exploration` explores the datasets;
`01`–`04` run the SenseFi model zoo per dataset and tabulate results;
`05_wimans_counting` compares WiMANS classification and regression.
Notebooks are paired `.py` (jupytext percent) + `.ipynb`. After editing a
`.py`, regenerate with `uv run jupytext --to ipynb notebooks/<name>.py`.

## Tests

```bash
uv run pytest              # includes real-data tests when data/ exists
uv run pytest -m "not data"  # library-only, no datasets needed
```