# rfsensing

`rfsensing` is the modelling and evaluation layer of the
**RF Sensing for Individual and Group Classification** project. It provides
reusable PyTorch/Lightning components for integrating WiFi Channel State
Information (CSI) datasets, training sensing models, and comparing evaluation
protocols across public benchmarks and future in-house SDR captures.

The project currently supports activity recognition, closed-set person
identification, open-set person re-identification, and people counting. Its longer-term focus is robust,
device-free person identification and group analysis under changes in subject,
day, room, and sensing hardware.

The full project plan lives at `../../MAIN_PROJECT.md` in the parent
RF Sensing workspace and includes the OpenCPI/USRP X310 capture work. This
repository focuses on datasets, models, training, and evaluation.

## What works today

| Area | Implemented capabilities |
|---|---|
| Tasks | Activity classification, closed-set person classification, open-set person re-identification with unknown rejection, group-size classification, scalar count regression |
| Datasets | UT_HAR, NTU-Fi HAR, NTU-Fi HumanID (closed-set and identity-disjoint Re-ID views), Widar, WiMANS, and generated synthetic data |
| Models | MLP, LeNet, LSTM/BiLSTM, ResNet-18, and ViT |
| Representations | Dataset-provided CSI amplitude and Widar BVP tensors; configurable WiMANS temporal pooling and normalization |
| Evaluation | Accuracy, rank-k accuracy, confusion matrices, count MAE, ±1-person accuracy, rounded regression accuracy, and gallery–probe retrieval/rejection metrics |
| Training | Task-aware Lightning modules, joint classification + batch-hard triplet Re-ID training, best-checkpoint restoration, TensorBoard logging, and model embeddings |

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
    "pad_side": "left",        # "left" or "right"
    "pooling": "mean",          # "mean" or "max"
    "normalization": "train",   # "train", "sample", or "none"
    "split_strategy": "group",
    "split_ratios": (0.70, 0.15, 0.15),
    "split_seed": 42,
    "environments": None,       # e.g. ("classroom",)
    "wifi_bands": None,         # e.g. ("2.4", "5")
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

Set `split_strategy="random"` for a count-stratified sample split when
comparing with less restrictive protocols. `environments` and `wifi_bands`
filter annotations before file validation and splitting.

### Open-set person re-identification

Closed-set classification, identity-disjoint Re-ID, and unknown rejection are
three distinct tasks. `ntu_fi_humanid` trains and tests on the same 14
subjects. `ntu_fi_humanid_reid` instead splits subjects by role — the default
7/2/1/3/1 protocol assigns 7 training, 2 validation-enrolled,
1 validation-unknown, 3 test-enrolled, and 1 test-unknown identities per
deterministic seed — so evaluation subjects are never seen in training.
Enrolled identities are matched by cosine similarity between probe and gallery
embeddings; probes of unknown identities must additionally be rejected when
their top gallery score falls below a threshold.

Training draws identity-balanced P×K batches (P identities × K samples) and
optimizes a joint objective: cross-entropy over the training identities plus a
batch-hard triplet loss on L2-normalized embeddings. The best checkpoint is
selected by validation mAP. Both rejection thresholds — the validation EER
point and the strictest threshold with validation FAR ≤ 5% — are calibrated
on validation scores only and applied unchanged to the test probes.

```python
from pathlib import Path

from rfsensing import data, models
from rfsensing.train import run_reid_repeats

data_root = Path("../../data")


def make_dm(seed):
    return data.build(
        "ntu_fi_humanid_reid",
        root=data_root,
        split_seed=seed,
        identities_per_batch=4,
        samples_per_identity=4,
    )


resnet_result = run_reid_repeats(
    lambda dm: models.build(
        "resnet18", in_shape=dm.sample_shape, num_classes=dm.output_dim
    ),
    make_dm,
    seeds=(42, 43, 44),
    max_epochs=50,
    name="reid-resnet18",
)

vit_result = run_reid_repeats(
    lambda dm: models.build(
        "vit",
        in_shape=dm.sample_shape,
        num_classes=dm.output_dim,
        patch_size=(38, 50),
    ),
    make_dm,
    seeds=(42, 43, 44),
    max_epochs=50,
    name="reid-vit",
)
print(vit_result.aggregate_metrics["test/mAP"])
```

Metric interpretation:

- **rank-1, rank-3, mAP** — retrieval quality over known probes only. Rank-3
  replaces the customary rank-5 because only three identities are enrolled in
  each test repeat, which would make rank-5 trivially perfect.
- **AUROC, EER** — threshold-free known-versus-unknown detection quality of
  the top gallery score.
- **DIR, FAR, unknown-rejection rate, known acceptance** — operating-point
  metrics at the validation-derived thresholds. DIR counts a known probe only
  when it is both accepted and assigned the correct top identity, so
  rejecting everything cannot look successful.

Each repeat saves its artifacts under
`runs/ntu_fi_humanid_reid/<name>/seed<seed>/version_<n>/`: the identity
manifest (`manifest.json`), run configuration (`config.json`), best
checkpoint, per-probe scores and thresholded predictions
(`predictions.csv`), and metrics with threshold provenance (`summary.json`).
`run_reid_repeats` writes mean ± sample standard deviation over seeds to
`runs/ntu_fi_humanid_reid/<name>/aggregate_summary.json`. A single repeat is
available as `rfsensing.train.run_reid`.

Training device selection follows Lightning's `accelerator` argument
(`"auto"` by default, so CUDA/MPS is used when available). Post-training
embedding extraction runs on the same device; pass `device=` to `run_reid`
or `run_reid_repeats` to override it (e.g. `device="cpu"`).

## Datasets

| Registry name | Task | Sample shape | Classes/output | Default protocol |
|---|---|---:|---:|---|
| `ut_har` | Activity classification | `(1, 250, 90)` | 7 | Fixed train/validation/test |
| `ntu_fi_har` | Activity classification | `(3, 114, 500)` | 6 | Fixed train/test; test also serves validation |
| `ntu_fi_humanid` | Closed-set person classification | `(3, 114, 500)` | 14 | Fixed train/test; test also serves validation |
| `ntu_fi_humanid_reid` | Open-set person re-identification | `(3, 114, 500)` | 7 train identities | Identity-disjoint 7/2/1/3/1 roles per seed |
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
| `06_open_set_person_reid` | NTU-Fi identity-disjoint Re-ID with unknown rejection |

Notebooks are paired Jupytext percent-format `.py` sources and generated
`.ipynb` files. Edit the Python source, then regenerate:

```bash
uv run jupytext --to ipynb notebooks/05_wimans_counting.py
```

## Research direction

The next steps align this package with the project’s individual and group
sensing goals:

- **WhoFi reproduction:** a faithful WhoFi architecture and
  published-protocol reproduction remains explicit future work; the current
  ViT Re-ID baseline is a generic Transformer, not a WhoFi implementation.
- **Richer Re-ID objectives:** ArcFace and supervised contrastive losses on
  top of the existing joint cross-entropy + triplet training.
- **Robustness protocols:** leave-one-day-out and leave-one-room-out splits
  instead of relying only on fixed or random splits.
- **Temporal gait models:** CNN+GRU and temporal Transformer baselines for
  person identification.
- **Richer CSI representations:** sanitized phase, antenna-ratio features, and
  Doppler/spectrogram views.
- **In-house SDR data:** an HDF5 DataModule for CSI exported by the
  OpenCPI/USRP X310 capture pipeline, including HDF5-backed Re-ID support.
- **Group structure:** counting baselines first, followed by coarse spatial
  configuration classification when suitable labels are available.

These are roadmap items, not implemented features. The sibling capture project
owns RF acquisition; `rfsensing` will consume its stable exported format.

## Extending `rfsensing`

Register a DataModule with:

```python
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from rfsensing import data
from rfsensing.data import CSIDataModule


@data.register("my_dataset")
class MyDataModule(CSIDataModule):
    name = "my_dataset"
    sample_shape = (3, 30, 100)
    class_names = ["class_0", "class_1"]

    def __init__(self, root, batch_size=64, num_workers=0):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.root = Path(root)

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(32, *self.sample_shape, generator=generator)
        y = torch.arange(32) % self.num_classes
        dataset = TensorDataset(x, y)
        self.train_set = dataset
        self.val_set = dataset
        self.test_set = dataset

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)
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
