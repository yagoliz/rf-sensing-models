# %% [markdown]
# # WhoFi reproduction on NTU-Fi HumanID
#
# WhoFi (Avola et al. 2025, arXiv:2507.12869) performs person
# re-identification from Wi-Fi CSI: a signal encoder (LSTM, BiLSTM, or
# Transformer) maps each amplitude sequence to an L2-normalized "signature"
# trained with an in-batch negative loss, and probes are matched to a
# gallery by cosine similarity. Its headline result is rank-1 0.955 / mAP
# 0.884 on NTU-Fi HumanID with a 1-layer Transformer.
#
# This notebook reproduces that benchmark with `models.build("whofi", ...)`
# and `run_whofi_repeats`, then evaluates the same architecture under this
# repo's identity-disjoint open-set protocol (notebooks 06/07) — the
# comparison the paper cannot make, since its protocol trains on all 14
# identities.
#
# ## Faithful vs. chosen vs. deviating
#
# Faithful to the paper: unfiltered amplitudes (their ablation found Hampel
# filtering *hurt*), sequences subsampled to 100 packets (their best packet
# length), a 1-layer Transformer with sinusoidal positional encodings,
# last-hidden-state readout for LSTM/BiLSTM, the in-batch negative loss over
# paired same-identity samples, Adam at lr 1e-4 with StepLR 0.95 every 50
# epochs, 8 identity pairs per batch, 300 epochs, and **no model
# selection** (the final model is evaluated).
#
# Chosen where the paper is silent: model width 128, 8 attention heads,
# feed-forward width 256, signature dimension 128, mean pooling over the
# Transformer output, dropout 0.1.
#
# Known deviations: (1) the SenseFi packaging of NTU-Fi already downsamples
# 2000 packets to 500 with fixed-constant normalization — WhoFi subsamples
# from the raw 2000; (2) no data augmentation (the paper reports no
# significant benefit for the Transformer; it did help their LSTMs, so the
# recurrent rows here are expected to lag their table); (3) three seeds
# instead of 3-fold cross-validation; (4) the paper does not specify how the
# evaluation gallery is built, so both readings are reported: `enrollment`
# (gallery = training samples, queries = test samples) and `loo`
# (leave-one-out within the test split).

# %%
import json
from pathlib import Path

import pandas as pd

from rfsensing import data, models
from rfsensing.train import run_reid_repeats, run_whofi_repeats

DATA_DIR = Path.cwd().resolve().parents[2] / "data"
RUNS_DIR = Path("runs")
SEEDS = (42, 43, 44)
EPOCHS = 300
ENCODERS = ("transformer", "lstm", "bilstm")
RUN_REPRODUCTION = False

PUBLISHED = pd.DataFrame(
    {
        "transformer": {
            "rank1": "0.955 ± 0.013",
            "rank3": "0.981 ± 0.006",
            "rank5": "0.991 ± 0.000",
            "mAP": "0.884 ± 0.012",
        },
        "bilstm": {
            "rank1": "0.845 ± 0.045",
            "rank3": "0.934 ± 0.022",
            "rank5": "0.958 ± 0.013",
            "mAP": "0.612 ± 0.026",
        },
        "lstm": {
            "rank1": "0.777 ± 0.032",
            "rank3": "0.897 ± 0.014",
            "rank5": "0.933 ± 0.005",
            "mAP": "0.568 ± 0.010",
        },
    }
).T

assert (DATA_DIR / "NTU-Fi-HumanID").is_dir(), (
    f"NTU-Fi-HumanID not found under {DATA_DIR}"
)

# %% [markdown]
# ## Closed-set reproduction
#
# `run_whofi` trains on the full training split (546 samples, all 14
# subjects) and evaluates retrieval on the 294 test samples. Each epoch
# re-reads the `.mat` files, so the three-encoder benchmark takes a while;
# it is opt-in like every training cell in this repo.

# %%
def make_dm(seed):
    return data.build("ntu_fi_humanid", root=DATA_DIR)


if RUN_REPRODUCTION:
    reproduction = {}
    for encoder in ENCODERS:
        reproduction[encoder] = run_whofi_repeats(
            lambda dm: models.build(
                "whofi",
                in_shape=(3, 114, 500),
                num_classes=14,
                encoder=encoder,
            ),
            make_dm,
            seeds=SEEDS,
            max_epochs=EPOCHS,
            name=f"whofi-{encoder}",
            runs_dir=RUNS_DIR,
        )

# %%
METRICS = ("rank1", "rank3", "rank5", "mAP")


def reproduction_table(protocol):
    rows = {}
    for encoder in ENCODERS:
        aggregate_path = (
            RUNS_DIR / "ntu_fi_humanid" / f"whofi-{encoder}"
            / "aggregate_summary.json"
        )
        if not aggregate_path.exists():
            print(f"no saved results for whofi-{encoder}; run training first")
            continue
        stats = json.loads(aggregate_path.read_text())["metrics"]
        rows[encoder] = {
            metric: (
                f"{stats[f'test/{protocol}/{metric}']['mean']:.3f} ± "
                f"{stats[f'test/{protocol}/{metric}']['std']:.3f}"
            )
            for metric in METRICS
        }
    return pd.DataFrame(rows).T


pd.concat(
    {"published": PUBLISHED, "enrollment": reproduction_table("enrollment")},
    axis=0,
)

# %%
pd.concat(
    {"published": PUBLISHED, "leave-one-out": reproduction_table("loo")},
    axis=0,
)

# %% [markdown]
# Reading the comparison:
#
# - **The qualitative claims reproduce.** The encoder ordering
#   LSTM < BiLSTM < Transformer holds under both gallery readings, and the
#   1-layer Transformer clears the published rank-1 0.955.
# - **Every row lands *above* its published number** — Transformer 0.988 vs.
#   0.955 rank-1 (mAP 0.988 vs. 0.884), BiLSTM 0.924 vs. 0.845, LSTM 0.872
#   vs. 0.777 — including the recurrent encoders, which here train without
#   the augmentation their ablation credits. The absolute numbers are
#   therefore not strictly comparable: the SenseFi packaging already
#   downsamples 2000 packets to 500 (a mild denoiser), this run trains on
#   the full training split where the paper's 3-fold protocol holds out 20%
#   per fold, and the gallery construction is our reading of an unspecified
#   protocol. The uplift is systematic, not a tuning fluke on one encoder.
# - Closed-set NTU-Fi HumanID is close to saturated (notebook 03's plain
#   classifiers also sit near ceiling), so the interesting question is not
#   this table but how the architecture behaves on identities it never saw.

# %% [markdown]
# ## WhoFi under the open-set protocol
#
# Same encoder, same loss — but trained on 7 identities and evaluated on
# disjoint ones under the notebook 06/07 protocol (8 seeds, 50 epochs,
# top-score detection, thresholds calibrated on validation). The in-batch
# negative loss slots into `ReIDModule` as `objective="inbatch"`; it needs
# exactly two samples per identity per batch.

# %%
RUN_OPEN_SET = False


def make_reid_dm(seed):
    return data.build(
        "ntu_fi_humanid_reid",
        root=DATA_DIR,
        split_seed=seed,
        identities_per_batch=7,
        samples_per_identity=2,
    )


if RUN_OPEN_SET:
    open_set = run_reid_repeats(
        lambda dm: models.build(
            "whofi", in_shape=dm.sample_shape, num_classes=dm.output_dim
        ),
        make_reid_dm,
        seeds=tuple(range(42, 50)),
        max_epochs=50,
        name="whofi-reid-transformer",
        runs_dir=RUNS_DIR,
        objective="inbatch",
        detection_score="top_score",
    )

# %%
OPEN_SET_ROWS = {
    "triplet / resnet18": "extended-reid-resnet18",
    "supcon / resnet18": "supcon-gap-reid-resnet18",
    "arcface / resnet18": "arcface-reid-resnet18",
    "triplet / vit": "extended-reid-vit",
    "whofi (inbatch / transformer)": "whofi-reid-transformer",
}
OPEN_SET_METRICS = [
    "test/rank1",
    "test/mAP",
    "test/auroc",
    "test/eer_threshold/dir",
    "test/far05_threshold/dir",
]


def open_set_table():
    rows = {}
    for label, prefix in OPEN_SET_ROWS.items():
        aggregate_path = (
            RUNS_DIR / "ntu_fi_humanid_reid" / prefix / "aggregate_summary.json"
        )
        if not aggregate_path.exists():
            print(f"no saved results for {prefix}; run training first")
            continue
        stats = json.loads(aggregate_path.read_text())["metrics"]
        rows[label] = {
            metric: f"{stats[metric]['mean']:.3f} ± {stats[metric]['std']:.3f}"
            for metric in OPEN_SET_METRICS
        }
    return pd.DataFrame(rows).T


open_set_table()

# %% [markdown]
# ## Conclusions and caveats
#
# - The WhoFi result reproduces — and then some: all three encoders exceed
#   their published numbers on the SenseFi packaging, with the paper's
#   encoder ordering intact. The published table should be read as
#   conservative for this data packaging rather than as a ceiling.
# - The paper's augmentation (Gaussian noise, scaling, time shift) is not
#   implemented, yet the recurrent rows still clear their published
#   numbers; on 500-packet SenseFi data it does not appear necessary.
# - Closed-set retrieval on 14 seen identities is nearly saturated and
#   should not be read as person re-identification working in the wild. The
#   open-set table is the honest benchmark: there the WhoFi encoder posts
#   the best raw retrieval of any variant so far (rank-1 0.959) but a
#   mid-pack operating point (DIR@EER 0.70 vs. SupCon/ResNet18's 0.81) —
#   a strong encoder does not automatically make a well-calibrated
#   open-set rejector.
# - Everything inherits NTU-Fi's limits (14 subjects, single room/day). The
#   next real test for any of these models is in-house SDR data with more
#   identities and session/room shift.
