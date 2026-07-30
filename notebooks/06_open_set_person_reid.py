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


def aggregate_table(metric_names, prefix="reid"):
    rows = {}
    for encoder in ENCODERS:
        summary_path = (
            RUNS_DIR / "ntu_fi_humanid_reid" / f"{prefix}-{encoder}"
            / "aggregate_summary.json"
        )
        if not summary_path.exists():
            print(f"no saved results for {prefix}-{encoder}; run training first")
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
def score_histogram(prefix="reid"):
    prediction_files = sorted(
        RUNS_DIR.glob(
            f"ntu_fi_humanid_reid/{prefix}-*/seed*/version_*/predictions.csv"
        )
    )
    if not prediction_files:
        print(f"no predictions saved under {prefix}-*; run training first")
        return
    predictions = pd.concat(pd.read_csv(p) for p in prediction_files)
    # Newer runs record the score actually thresholded (top_score or gap);
    # fall back to top_score for CSVs from before that column existed.
    column = (
        "detection_score" if "detection_score" in predictions else "top_score"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for flag, label in ((True, "known probes"), (False, "unknown probes")):
        subset = predictions[predictions["known"] == flag]
        ax.hist(subset[column], bins=30, alpha=0.6, label=label)
    ax.set(xlabel=f"detection score ({column})", ylabel="probes")
    ax.legend(frameon=False)


score_histogram()

# %% [markdown]
# ## Extended benchmark: more seeds, longer training
#
# In short runs the open-set numbers swing wildly between seeds because each
# repeat's rejection statistics hinge on a **single** unknown test subject.
# More repeats tighten the mean ± std (every seed rotates different subjects
# through the enrolled/unknown roles), and more epochs mainly help the ViT,
# which lags the ResNet in short runs.
#
# This section is separately opt-in and writes under `extended-reid-*`, so it
# never overwrites the quick-run artifacts above. Expect it to take roughly
# `len(EXTENDED_SEEDS) * EXTENDED_EPOCHS / (len(SEEDS) * EPOCHS)` times the
# short benchmark.

# %%
RUN_EXTENDED = False
EXTENDED_SEEDS = tuple(range(42, 50))  # 8 identity-role rotations
EXTENDED_EPOCHS = 50

if RUN_EXTENDED:
    extended = {}
    for encoder, kwargs in ENCODERS.items():
        extended[encoder] = run_reid_repeats(
            lambda dm: models.build(
                encoder,
                in_shape=dm.sample_shape,
                num_classes=dm.output_dim,
                **kwargs,
            ),
            make_dm,
            seeds=EXTENDED_SEEDS,
            max_epochs=EXTENDED_EPOCHS,
            name=f"extended-reid-{encoder}",
            runs_dir=RUNS_DIR,
        )

# %%
aggregate_table(RETRIEVAL, prefix="extended-reid")

# %%
aggregate_table(OPEN_SET, prefix="extended-reid")

# %% [markdown]
# When comparing the extended table with the quick one, look at the std
# columns first: retrieval metrics should be stable, while the
# `far05_threshold` operating point keeps a large spread — that residual
# variance comes from calibrating FAR on one unknown validation subject
# (~39 probes), a limit of the 14-subject dataset rather than of training.

# %%
score_histogram("extended-reid")

# %% [markdown]
# ## Variant: SupCon objective + top-gap detection score
#
# Two options aimed at the failure modes of absolute-score thresholding:
#
# - `objective="supcon"` replaces the batch-hard triplet term with a
#   supervised contrastive loss whose log-sum-exp keeps pushing all
#   negatives apart, spreading identities over the hypersphere instead of
#   stopping at a fixed margin — cosine scores stop saturating near 1.0.
# - `detection_score="top_gap"` rejects on the top-1 minus top-2 identity
#   score instead of the absolute top cosine. A probe equidistant from two
#   enrolled identities has a high top score but a near-zero gap, so gap
#   thresholds are robust to per-subject score shifts.
#
# Both are independent switches; this cell runs them together on the same
# seeds as the extended benchmark for a paired comparison.

# %%
RUN_VARIANT = False

if RUN_VARIANT:
    variant = {}
    for encoder, kwargs in ENCODERS.items():
        variant[encoder] = run_reid_repeats(
            lambda dm: models.build(
                encoder,
                in_shape=dm.sample_shape,
                num_classes=dm.output_dim,
                **kwargs,
            ),
            make_dm,
            seeds=EXTENDED_SEEDS,
            max_epochs=EXTENDED_EPOCHS,
            name=f"supcon-gap-reid-{encoder}",
            runs_dir=RUNS_DIR,
            objective="supcon",
            detection_score="top_gap",
        )

# %%
aggregate_table(RETRIEVAL, prefix="supcon-gap-reid")

# %%
aggregate_table(OPEN_SET, prefix="supcon-gap-reid")

# %%
score_histogram("supcon-gap-reid")

# %% [markdown]
# Judge the variant against `extended-reid` per seed, not only on the means:
# the interesting questions are whether the reject-all rotations (calibrated
# thresholds near 1.0) recover DIR, and whether the confusable-unknown
# rotation improves at all — if that subject genuinely resembles an enrolled
# one, no objective can separate them and only more training identities help.

# %% [markdown]
# ## Embedding space per individual: triplet vs. SupCon
#
# Two paired views of one seed's **test** roles, computed from the saved best
# checkpoints of both objectives (no retraining):
#
# 1. a 2-D t-SNE projection colored by subject, with gallery samples, known
#    probes, and unknown probes drawn as distinct markers — cluster overlap
#    with the unknown subject is the "confusable rotation" failure made
#    visible, and gallery/probe drift within one color shows session shift;
# 2. intra- vs. inter-identity cosine similarity histograms — the faithful
#    view of the *spread* claim. t-SNE exaggerates cluster separation and
#    discards absolute distances, so panels can look alike even when the
#    score geometry differs; the histograms show the quantity thresholds
#    actually operate on. Expect SupCon to shift the inter-identity mass
#    away from 1.0 and widen the gap between the two distributions.
#
# Set `VIZ_SEED` to the confusable rotation (test-unknown subject scoring
# like an enrolled one) to see failure mode 1 directly.

# %%
import numpy as np
import torch
import torch.nn.functional as F

from rfsensing.eval.embeddings import project

VIZ_SEED = EXTENDED_SEEDS[0]
VIZ_ENCODER = "resnet18"
OBJECTIVE_RUNS = {"triplet": "extended-reid", "supcon": "supcon-gap-reid"}


def load_run_net(prefix, encoder, seed):
    run_root = (
        RUNS_DIR / "ntu_fi_humanid_reid" / f"{prefix}-{encoder}" / f"seed{seed}"
    )
    summaries = sorted(run_root.glob("version_*/summary.json"))
    if not summaries:
        print(f"no saved run under {run_root}; train that variant first")
        return None
    summary = json.loads(summaries[-1].read_text())
    net = models.build(
        encoder,
        in_shape=viz_dm.sample_shape,
        num_classes=viz_dm.output_dim,
        **ENCODERS[encoder],
    )
    state = torch.load(
        summary["checkpoint"], map_location="cpu", weights_only=True
    )["state_dict"]
    net.load_state_dict(
        {k.removeprefix("net."): v for k, v in state.items() if k.startswith("net.")}
    )
    return net.eval()


@torch.no_grad()
def embed_test_roles(net, dm):
    roles = {}
    for role, loader in dm.test_loaders_by_role().items():
        zs, ys = [], []
        for x, y in loader:
            zs.append(F.normalize(net.embed(x), dim=1))
            ys.append(y)
        roles[role] = (torch.cat(zs).numpy(), torch.cat(ys).numpy())
    return roles


viz_dm = make_dm(VIZ_SEED)
viz_dm.setup()
embedded = {}
for objective, prefix in OBJECTIVE_RUNS.items():
    net = load_run_net(prefix, VIZ_ENCODER, VIZ_SEED)
    if net is not None:
        embedded[objective] = embed_test_roles(net, viz_dm)

# %%
ROLE_MARKERS = {"gallery": "o", "known_probes": "^", "unknown_probes": "x"}

if embedded:
    fig, axes = plt.subplots(
        1, len(embedded), figsize=(6 * len(embedded), 5), squeeze=False
    )
    for ax, (objective, roles) in zip(axes[0], embedded.items()):
        z_all = np.concatenate([z for z, _ in roles.values()])
        z2d = project(z_all, method="tsne", seed=VIZ_SEED)
        subjects = sorted(
            {int(s) for _, y in roles.values() for s in np.unique(y)}
        )
        colors = {s: plt.get_cmap("tab10")(i) for i, s in enumerate(subjects)}
        offset = 0
        for role, (z, y) in roles.items():
            block = z2d[offset : offset + len(z)]
            offset += len(z)
            for subject in np.unique(y):
                mask = y == subject
                unknown = role == "unknown_probes"
                ax.scatter(
                    block[mask, 0],
                    block[mask, 1],
                    marker=ROLE_MARKERS[role],
                    s=60 if unknown else 30,
                    color=colors[int(subject)],
                    linewidths=1.5 if unknown else 0.5,
                    label=(
                        f"{viz_dm.identity_names[int(subject)]}"
                        f" ({role.replace('_', ' ')})"
                    ),
                )
        ax.set(title=f"{VIZ_ENCODER}, {objective}, seed {VIZ_SEED}")
        ax.set_xticks([])  # t-SNE axes are unitless
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()

# %%
if embedded:
    fig, axes = plt.subplots(
        1, len(embedded), figsize=(6 * len(embedded), 4),
        squeeze=False, sharex=True,
    )
    bins = np.linspace(-0.2, 1.0, 61)
    for ax, (objective, roles) in zip(axes[0], embedded.items()):
        z = np.concatenate([z for z, _ in roles.values()])
        y = np.concatenate([y for _, y in roles.values()])
        similarities = z @ z.T
        same = y[:, None] == y[None, :]
        upper = np.triu(np.ones_like(same, dtype=bool), k=1)
        intra = similarities[same & upper]
        inter = similarities[~same & upper]
        ax.hist(intra, bins=bins, alpha=0.6, density=True, label="intra-identity")
        ax.hist(inter, bins=bins, alpha=0.6, density=True, label="inter-identity")
        ax.set(
            title=f"{objective}: cosine similarity",
            xlabel="pairwise cosine similarity",
            ylabel="density",
        )
        ax.legend(frameon=False)
    fig.tight_layout()

# %% [markdown]
# ## Caveats
#
# - The quick runs above are a pipeline **smoke test**, not paper-comparable
#   results: epochs, tuning, and repeats are all minimal. The extended
#   benchmark is closer to reportable, but still inherits NTU-Fi's limits
#   (one unknown subject per repeat).
# - Rank-3 replaces rank-5 because only three identities are enrolled per
#   test repeat — and with exactly three enrolled, rank-3 is trivially 1.0;
#   it only becomes informative with larger galleries.
# - WhoFi-style architectures, ArcFace/supervised-contrastive objectives,
#   and leave-one-day/room-out protocols are future work tracked in the
#   README roadmap.
