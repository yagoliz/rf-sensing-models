# %% [markdown]
# # Open-set person re-identification on NTU-Fi HumanID
#
# Closed-set HumanID classification (notebook 03) assumes every test subject
# was seen in training. This notebook instead trains identity-disjoint Re-ID
# embeddings: the encoder never sees the evaluation subjects, probes are
# matched against a gallery by cosine similarity, and probes of subjects
# absent from the gallery must be rejected as unknown.
#
# Protocol per repeat (14 subjects, deterministic per seed): 7 train,
# 2 validation-enrolled, 1 validation-unknown, 3 test-enrolled,
# 1 test-unknown. Thresholds are calibrated on validation scores only.
#
# The ViT here is the generic rfsensing baseline — it is **not** a WhoFi
# implementation; a faithful WhoFi reproduction remains future work.

# %%
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rfsensing import data, models
from rfsensing.train import run_reid, run_reid_repeats

DATA_DIR = Path.cwd().resolve().parents[2] / "data"
RUNS_DIR = Path("runs")
SEEDS = (42, 43, 44)
EPOCHS = 15
# Training three repeats per encoder takes a while on CPU; keep it opt-in.
RUN_TRAINING = False

assert (DATA_DIR / "NTU-Fi-HumanID").is_dir(), (
    f"NTU-Fi-HumanID not found under {DATA_DIR}"
)

# %% [markdown]
# ## Identity roles and P×K batches
#
# Each repeat's manifest assigns every subject exactly one role. Training
# batches are identity-balanced: P identities × K samples, the shape the
# batch-hard triplet loss needs.

# %%
dm = data.build(
    "ntu_fi_humanid_reid",
    root=DATA_DIR,
    split_seed=SEEDS[0],
    identities_per_batch=4,
    samples_per_identity=4,
)
print(json.dumps(dm.split_manifest.to_dict(), indent=2))

# %%
dm.setup("fit")
x_batch, y_batch = next(iter(dm.train_dataloader()))
print("batch:", tuple(x_batch.shape), "labels:", Counter(y_batch.tolist()))

# %% [markdown]
# ## Repeated experiments
#
# `run_reid_repeats` trains one repeat per seed (fresh split, fresh network),
# calibrates EER and FAR≤5% thresholds on validation, evaluates them
# unchanged on test probes, and saves all artifacts under `runs/`.

# %%
ENCODERS = {
    "resnet18": {"base_width": 32},
    "vit": {"patch_size": (38, 50), "embed_dim": 64, "depth": 2},
}


def make_dm(seed):
    return data.build(
        "ntu_fi_humanid_reid",
        root=DATA_DIR,
        split_seed=seed,
        identities_per_batch=4,
        samples_per_identity=4,
    )


if RUN_TRAINING:
    repeated = {}
    for encoder, kwargs in ENCODERS.items():
        repeated[encoder] = run_reid_repeats(
            lambda dm: models.build(
                encoder,
                in_shape=dm.sample_shape,
                num_classes=dm.output_dim,
                **kwargs,
            ),
            make_dm,
            seeds=SEEDS,
            max_epochs=EPOCHS,
            name=f"reid-{encoder}",
            runs_dir=RUNS_DIR,
        )

# %% [markdown]
# A single repeat is the same call without the factories — useful when
# iterating on one architecture before committing to the full benchmark.

# %%
if RUN_TRAINING:
    single = run_reid(
        models.build(
            "resnet18",
            in_shape=dm.sample_shape,
            num_classes=dm.output_dim,
            base_width=32,
        ),
        dm,
        max_epochs=EPOCHS,
        name="reid-resnet18-single",
        seed=SEEDS[0],
        runs_dir=RUNS_DIR,
    )
    print(single.thresholds)

# %% [markdown]
# ## Results from saved artifacts
#
# The saved JSON/CSV artifacts are the source of truth; the tables below are
# presentations of those files and work in a fresh session once training has
# run at least once.

# %%
RETRIEVAL = ["test/rank1", "test/rank3", "test/mAP", "test/auroc", "test/eer"]
OPEN_SET = [
    "test/eer_threshold/dir",
    "test/eer_threshold/far",
    "test/far05_threshold/dir",
    "test/far05_threshold/far",
    "test/far05_threshold/unknown_rejection",
]


def aggregate_table(metric_names):
    rows = {}
    for encoder in ENCODERS:
        summary_path = (
            RUNS_DIR / "ntu_fi_humanid_reid" / f"reid-{encoder}"
            / "aggregate_summary.json"
        )
        if not summary_path.exists():
            print(f"no saved results for {encoder}; set RUN_TRAINING = True")
            continue
        stats = json.loads(summary_path.read_text())["metrics"]
        rows[encoder] = {
            name: f"{stats[name]['mean']:.3f} ± {stats[name]['std']:.3f}"
            for name in metric_names
        }
    return pd.DataFrame(rows).T


aggregate_table(RETRIEVAL)

# %%
aggregate_table(OPEN_SET)

# %% [markdown]
# ## Known vs. unknown score distributions
#
# Rejection quality is visible directly in the per-probe top cosine scores:
# enrolled (known) probes should score higher than unknown probes.

# %%
prediction_files = sorted(
    RUNS_DIR.glob("ntu_fi_humanid_reid/reid-*/seed*/version_*/predictions.csv")
)
if prediction_files:
    predictions = pd.concat(pd.read_csv(p) for p in prediction_files)
    fig, ax = plt.subplots(figsize=(7, 4))
    for flag, label in ((True, "known probes"), (False, "unknown probes")):
        subset = predictions[predictions["known"] == flag]
        ax.hist(subset["top_score"], bins=30, alpha=0.6, label=label)
    ax.set(xlabel="top gallery cosine score", ylabel="probes")
    ax.legend(frameon=False)
else:
    print("no predictions saved yet; set RUN_TRAINING = True")

# %% [markdown]
# ## Caveats
#
# - These short runs are a pipeline **smoke test**, not paper-comparable
#   results: epochs, tuning, and repeats are all minimal.
# - Rank-3 replaces rank-5 because only three identities are enrolled per
#   test repeat.
# - WhoFi-style architectures, ArcFace/supervised-contrastive objectives,
#   and leave-one-day/room-out protocols are future work tracked in the
#   README roadmap.
