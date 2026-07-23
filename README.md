# rfsensing

`rfsensing` is the modelling and evaluation layer of the
**RF Sensing for Individual and Group Classification** project. It provides
reusable PyTorch/Lightning components for integrating WiFi Channel State
Information (CSI) datasets, training sensing models, and comparing evaluation
protocols across public benchmarks and future in-house SDR captures.

The project currently supports activity recognition, closed-set person
identification, and people counting. Its longer-term focus is robust,
device-free person identification and group analysis under changes in subject,
day, room, and sensing hardware.

See [`../../MAIN_PROJECT.md`](../../MAIN_PROJECT.md) for the full project,
including the OpenCPI/USRP X310 capture work. This repository focuses on
datasets, models, training, and evaluation.

## What works today

| Area | Implemented capabilities |
|---|---|
| Tasks | Activity classification, closed-set person classification, group-size classification, scalar count regression |
| Datasets | UT_HAR, NTU-Fi HAR, NTU-Fi HumanID, Widar, WiMANS, and generated synthetic data |
| Models | MLP, LeNet, LSTM/BiLSTM, ResNet-18, and ViT |
| Representations | Dataset-provided CSI amplitude tensors; configurable WiMANS temporal pooling and normalization |
| Evaluation | Accuracy, rank-k accuracy, confusion matrices, count MAE, ±1-person accuracy, and rounded regression accuracy |
| Training | Task-aware Lightning modules, best-checkpoint restoration, TensorBoard logging, and model embeddings |

All DataModules and models share registry-based interfaces, so experiments are
constructed from dataset metadata rather than hard-coded tensor dimensions.

## Installation

```bash
uv sync
```

Run commands through the managed environment:

```bash
uv run pytest -m "not data"
uv run jupyter lab
uv run tensorboard --logdir runs
```

## Quick start

### Classification

```python
from pathlib import Path

from rfsensing import data, models, train

data_root = Path("../../data")
dm = data.build("ntu_fi_humanid", root=data_root, batch_size=64)
net = models.build(
    "vit",
    in_shape=dm.sample_shape,
    num_classes=dm.output_dim,
)

result = train.run(net, dm, max_epochs=50, name="vit-humanid")
print(result.metrics)
best_net = train.load_best_net(net, result)
x, _ = next(iter(dm.test_dataloader()))
device = next(best_net.parameters()).device
embeddings = best_net.embed(x.to(device))
```

Existing classification DataModules use cross-entropy training and select the
best checkpoint by validation accuracy.

### WiMANS people counting

WiMANS supports two views of group size:

- ordered classification over counts 0–5; and
- scalar regression with raw MAE.

```python
common = {
    "root": data_root,
    "time_steps": 300,
    "pooling": "mean",          # "mean" or "max"
    "normalization": "train",   # "train", "sample", or "none"
    "split_strategy": "group",
    "split_ratios": (0.70, 0.15, 0.15),
    "split_seed": 42,
    "batch_size": 32,
}

dm_class = data.build("wimans", target="classification", **common)
class_net = models.build(
    "lstm",
    in_shape=dm_class.sample_shape,
    num_classes=dm_class.output_dim,
    seq_axis=2,
)
class_result = train.run(
    class_net, dm_class, max_epochs=30, name="wimans-classification"
)

dm_reg = data.build("wimans", target="regression", **common)
reg_net = models.build(
    "lstm",
    in_shape=dm_reg.sample_shape,
    num_classes=dm_reg.output_dim,
    seq_axis=2,
)
reg_result = train.run(
    reg_net, dm_reg, max_epochs=30, name="wimans-regression"
)
```

The default group-held-out split keeps every `act_<group>_*` combination in
exactly one split. This is more resistant to repetition leakage than a random
sample split. Classification reports exact accuracy, MAE, and ±1-person
accuracy. Regression reports raw MAE, ±1-person accuracy, and rounded exact
accuracy, and checkpoints on minimum validation MAE.

## Datasets

| Registry name | Task | Sample shape | Classes/output | Default protocol |
|---|---|---:|---:|---|
| `ut_har` | Activity classification | `(1, 250, 90)` | 7 | Fixed train/validation/test |
| `ntu_fi_har` | Activity classification | `(3, 114, 500)` | 6 | Fixed train/test; test also serves validation |
| `ntu_fi_humanid` | Closed-set person classification | `(3, 114, 500)` | 14 | Fixed train/test; test also serves validation |
| `widar` | Gesture classification | `(22, 20, 20)` | 22 | Fixed train/test; test also serves validation |
| `wimans` | People-count classification or regression | `(9, 30, 300)` by default | 6 or 1 | Deterministic group-held-out 70/15/15 |
| `synthetic` | Test/smoke classification | Configurable | Configurable | Generated split |

Datasets live outside the repository at `../../data/`. Tests can use a
different root through `RFSENSING_DATA`, and callers can always pass `root=`
explicitly.

```text
../../data/
├── UT_HAR/
├── NTU-Fi_HAR/
├── NTU-Fi-HumanID/
├── Widardata/
└── WiMANS/
    ├── annotation.csv
    └── wifi_csi/amp/*.npy
```

WiMANS also accepts the flat `WiMANS/amp/*.npy` layout. Full training requires
all amplitude files selected by the annotations and optional environment/band
filters. `allow_partial=True` exists for loader exploration, but partial data
may not contain enough count groups for the default three-way split.

## Notebooks

The notebooks are experiment entry points rather than package internals:

| Notebook | Purpose |
|---|---|
| `00_data_exploration` | Inspect dataset metadata, shapes, and representative samples |
| `01_ut_har` | UT_HAR model comparison |
| `02_ntu_fi_har` | NTU-Fi activity-recognition benchmark |
| `03_ntu_fi_humanid` | NTU-Fi closed-set person-identification benchmark |
| `04_widar` | Widar gesture-recognition benchmark |
| `05_wimans_counting` | WiMANS count classification versus regression |

Notebooks are paired Jupytext percent-format `.py` sources and generated
`.ipynb` files. Edit the Python source, then regenerate:

```bash
uv run jupytext --to ipynb notebooks/05_wimans_counting.py
```

## Research direction

The next steps align this package with the project’s individual and group
sensing goals:

- **Open-set person identification:** metric-learning objectives and
  gallery–probe rank-1/rank-5/mAP evaluation.
- **Robustness protocols:** leave-one-day-out and leave-one-room-out splits
  instead of relying only on fixed or random splits.
- **Temporal gait models:** CNN+GRU and temporal Transformer baselines for
  person identification.
- **Richer CSI representations:** sanitized phase, antenna-ratio features, and
  Doppler/spectrogram views.
- **In-house SDR data:** an HDF5 DataModule for CSI exported by the
  OpenCPI/USRP X310 capture pipeline.
- **Group structure:** counting baselines first, followed by coarse spatial
  configuration classification when suitable labels are available.

These are roadmap items, not implemented features. The sibling capture project
owns RF acquisition; `rfsensing` will consume its stable exported format.

## Extending `rfsensing`

Register a DataModule with:

```python
from rfsensing import data
from rfsensing.data import CSIDataModule


@data.register("my_dataset")
class MyDataModule(CSIDataModule):
    name = "my_dataset"
    sample_shape = (3, 30, 100)
    class_names = ["class_0", "class_1"]
```

A sample is `(x, y)`: `x` is a `float32` tensor shaped as `sample_shape`;
classification targets are `int64` class indices and regression targets are
`float32` scalars. DataModules expose task metadata including `output_dim`,
`task_type`, and checkpoint monitor settings.

Register a model with:

```python
import math

import torch.nn as nn

from rfsensing import models


@models.register("my_model")
class MyModel(nn.Module):
    def __init__(self, in_shape, num_classes, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(in_shape), 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, num_classes)

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.embed(x))
```

Models are built with dataset-derived `in_shape` and `num_classes`. Every
built-in model exposes `embed(x)` and a final `head`.

## Development and testing

The full external datasets are not required for library development:

```bash
uv run pytest -m "not data"
```

Run data-marked integration tests when the corresponding complete datasets are
available:

```bash
uv run pytest
```

Training outputs and TensorBoard logs are written below
`runs/<dataset>/<experiment>/`.

## Origins and compatibility

The initial datasets, model zoo, and benchmark conventions were implemented as
a clean PyTorch/Lightning reworking of
[SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark). The project
retains those baselines for reproducibility while expanding toward additional
datasets, count regression, leakage-resistant splits, open-set identification,
robustness evaluation, and in-house SDR CSI.
