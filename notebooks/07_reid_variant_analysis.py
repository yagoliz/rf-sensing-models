# %% [markdown]
# # Re-ID variant analysis: per-seed behavior, detection scores, and ArcFace
#
# Notebook 06 established the open-set Re-ID benchmark and left three loose
# ends, which this notebook closes without retraining any of its runs:
#
# 1. **Per-seed paired comparison.** The aggregate tables showed SupCon
#    lifting mean DIR, but the interesting question is *where*: do the
#    reject-all rotations recover, and does the confusable-unknown rotation
#    improve at all?
# 2. **Detection-score forensics.** An audit of the saved artifacts shows
#    that *both* committed aggregates (`extended-reid-*` and
#    `supcon-gap-reid-*`) threshold the absolute top cosine score — the
#    superseded top-gap runs sit next to them as earlier `version_*` dirs.
#    This section shows why they were superseded: on this protocol the gap
#    score is not merely weaker, it is *anti-correlated* with being known.
# 3. **ArcFace.** The additive-angular-margin softmax
#    (`objective="arcface"`) moves from the roadmap into the benchmark.
#
# Protocol, seeds, and encoders are identical to notebook 06; everything here
# reads the artifacts under `runs/` plus the new `arcface-reid-*` runs.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rfsensing import data, models
from rfsensing.train import run_reid_repeats

DATA_DIR = Path.cwd().resolve().parents[2] / "data"
RUNS_DIR = Path("runs")
REID_RUNS = RUNS_DIR / "ntu_fi_humanid_reid"
SEEDS = tuple(range(42, 50))
EPOCHS = 50

ENCODERS = {
    "resnet18": {"base_width": 32},
    "vit": {"patch_size": (38, 50), "embed_dim": 64, "depth": 2},
}
# Variant name -> artifact prefix. All three committed benchmarks threshold
# the absolute top score; "supcon-gap" is a historical directory name from
# the superseded top-gap experiment (see the forensics section).
VARIANTS = {
    "triplet": "extended-reid",
    "supcon": "supcon-gap-reid",
    "arcface": "arcface-reid",
}

# %% [markdown]
# ## What the saved artifacts actually contain
#
# Every repeat stores its full configuration; trusting directory names is not
# necessary. This audit prints, for each aggregate, the (objective,
# detection score) pairs of the repeats it references — plus any stray
# `version_*` runs sitting in the same directories that the aggregate does
# **not** include.

# %%
def audit_variant(prefix, encoder):
    root = REID_RUNS / f"{prefix}-{encoder}"
    aggregate_path = root / "aggregate_summary.json"
    if not aggregate_path.exists():
        return None
    aggregate = json.loads(aggregate_path.read_text())
    referenced = {Path(r["summary"]).parent for r in aggregate["repeats"]}
    configs = {
        (c["objective"], c.get("detection_score", "top_score"))
        for run_dir in referenced
        for c in [json.loads((run_dir / "config.json").read_text())]
    }
    strays = {
        version
        for seed_dir in root.glob("seed*")
        for version in seed_dir.glob("version_*")
        if version not in referenced
    }
    return {"aggregated": sorted(configs), "stray_runs": len(strays)}


pd.DataFrame(
    {
        f"{name}/{encoder}": audit_variant(prefix, encoder)
        for name, prefix in VARIANTS.items()
        for encoder in ENCODERS
    }
).T

# %% [markdown]
# ## Per-seed paired comparison: triplet vs. SupCon
#
# Both committed aggregates use the same seeds, and each seed fixes the same
# identity-role rotation for every variant — so per-seed differences are
# paired observations, far more informative than the aggregate mean ± std
# (which is dominated by rotation difficulty, not by the objective).

# %%
METRICS = {
    "auroc": "test/auroc",
    "dir_eer": "test/eer_threshold/dir",
    "far_eer": "test/eer_threshold/far",
    "unknown_rejection_eer": "test/eer_threshold/unknown_rejection",
    "dir_far05": "test/far05_threshold/dir",
}


def per_seed_table(prefix, encoder):
    aggregate_path = REID_RUNS / f"{prefix}-{encoder}" / "aggregate_summary.json"
    if not aggregate_path.exists():
        print(f"no saved results for {prefix}-{encoder}; run training first")
        return None
    aggregate = json.loads(aggregate_path.read_text())
    rows = {}
    for repeat in aggregate["repeats"]:
        summary = json.loads(Path(repeat["summary"]).read_text())
        rows[repeat["seed"]] = {
            name: summary["metrics"][key] for name, key in METRICS.items()
        }
    return pd.DataFrame(rows).T.sort_index()


per_seed = {
    encoder: {
        name: per_seed_table(prefix, encoder)
        for name, prefix in VARIANTS.items()
    }
    for encoder in ENCODERS
}
per_seed["resnet18"]["triplet"].round(3)

# %%
per_seed["resnet18"]["supcon"].round(3)

# %%
def slope_plot(tables, title, metric="dir_eer"):
    tables = {name: table for name, table in tables.items() if table is not None}
    if not tables:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(tables)
    for seed in tables[names[0]].index:
        values = [tables[name].loc[seed, metric] for name in names]
        ax.plot(names, values, marker="o", alpha=0.7, label=f"seed {seed}")
    ax.set(
        title=f"{title}: per-rotation {metric} (paired seeds)",
        ylabel=metric,
        ylim=(-0.05, 1.05),
    )
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()


TRIPLET_VS_SUPCON = ("triplet", "supcon")
slope_plot(
    {name: per_seed["resnet18"][name] for name in TRIPLET_VS_SUPCON},
    "resnet18",
)

# %%
slope_plot(
    {name: per_seed["vit"][name] for name in TRIPLET_VS_SUPCON}, "vit"
)

# %% [markdown]
# Reading the paired ResNet18 numbers:
#
# - **The reject-all rotations recover.** Under triplet, seeds 43 and 45
#   calibrated their FAR≤5% threshold at 1.0 and rejected (nearly)
#   everything: DIR@EER 0.37 and 0.11. SupCon lifts exactly those two
#   rotations to 0.65 and 0.59 — the two largest paired deltas (+0.28,
#   +0.48). Rotations that were already healthy (46, 47, 49) move by ±0.02.
#   The aggregate improvement (0.69 → 0.81 mean DIR@EER) is therefore not a
#   uniform shift; it comes almost entirely from un-collapsing the failure
#   rotations.
# - **The confusable rotation is encoder-dependent.** Seed 48 is the
#   rotation whose unknown subject scores like an enrolled one: under
#   triplet the ResNet accepts the unknown subject at the EER point in 100%
#   of probes (unknown rejection 0.0, AUROC 0.85). SupCon *fixes* it for the
#   ResNet (AUROC 0.99, unknown rejection 1.0). The ViT does not recover
#   (AUROC 0.50 → 0.35): with the weaker encoder that subject remains
#   genuinely inseparable, which supports notebook 06's conjecture that only
#   more training identities — not the objective — help in that regime.
# - **SupCon is not free.** Seed 44 regresses (DIR@EER 0.80 → 0.74,
#   DIR@FAR05 0.74 → 0.54): spreading identities across the hypersphere can
#   cost a rotation that the margin objective already handled.

# %% [markdown]
# ## Detection-score forensics: top score vs. top gap
#
# The superseded runs are still on disk, and every `predictions.csv` from a
# top-gap run stores **both** scores per probe — enough to compare the two
# scorers on identical embeddings, no retraining needed. AUROC below 0.5
# means the scorer ranks unknown probes *above* known ones.

# %%
def scorer_comparison():
    rows = []
    for run_dir in sorted(REID_RUNS.glob("*/seed*/version_*")):
        config_path = run_dir / "config.json"
        predictions_path = run_dir / "predictions.csv"
        if not (config_path.exists() and predictions_path.exists()):
            continue
        config = json.loads(config_path.read_text())
        # Configs from before the detection_score switch thresholded the
        # absolute top score.
        if config.get("detection_score", "top_score") != "top_gap":
            continue  # top-score runs do not preserve the gap
        predictions = pd.read_csv(predictions_path)
        if predictions["known"].nunique() < 2:
            continue
        rows.append(
            {
                "variant": run_dir.parts[-3],
                "seed": run_dir.parts[-2],
                "objective": config["objective"],
                "auroc_gap": roc_auc_score(
                    predictions["known"], predictions["detection_score"]
                ),
                "auroc_top": roc_auc_score(
                    predictions["known"], predictions["top_score"]
                ),
            }
        )
    return pd.DataFrame(rows)


scorers = scorer_comparison()
if scorers.empty:
    print("no top-gap runs found on disk")
else:
    display(
        scorers.groupby(["variant", "objective"])[["auroc_gap", "auroc_top"]]
        .agg(["mean", "min"])
        .round(3)
    )

# %%
if not scorers.empty:
    fig, ax = plt.subplots(figsize=(5, 5))
    for objective, subset in scorers.groupby("objective"):
        ax.scatter(
            subset["auroc_top"], subset["auroc_gap"], label=objective, alpha=0.8
        )
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="gray", linewidth=0.5)
    ax.set(
        xlabel="AUROC with top score",
        ylabel="AUROC with top gap",
        title="Same embeddings, two rejection scores",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False)
    fig.tight_layout()

# %% [markdown]
# The same embeddings that give ~0.95 AUROC under the top score average
# ~0.4 — worse than chance — under the gap, and the ArcFace runs sink close
# to 0. The gap heuristic assumes an impostor sits *between* enrolled
# identities (high top score, near-zero lead over the runner-up). On this
# protocol the geometry is the opposite, as notebook 06's cosine histograms
# already showed: all embeddings live in a narrow cone, so a **genuine**
# probe scores ~0.99 against its identity while the runner-up identity is
# also high (~0.95) — a tiny gap — whereas an **unknown** probe sits away
# from all three enrolled clusters, where its per-identity scores are more
# spread and its lead is larger. The tighter the objective pulls identity
# clusters (ArcFace most of all), the more anti-correlated the gap becomes.
# With three-identity galleries and saturated cosines, `top_gap` is the
# wrong rejection score; it may still make sense for large galleries where
# impostor-between-identities is the dominant failure mode.

# %% [markdown]
# ## ArcFace: additive angular margin
#
# `objective="arcface"` replaces the joint CE + metric-loss objective with a
# margin softmax over the embeddings: logits are scaled cosines between the
# embedding and per-identity weight vectors, and the target identity's angle
# is penalized by `arcface_margin` radians before scaling by
# `arcface_scale`. The margin therefore shapes exactly the cosine geometry
# that gallery matching uses. Per the forensics above it is paired with the
# top-score detector (its tight clusters make the gap score maximally
# anti-correlated).

# %%
RUN_ARCFACE = False


def make_dm(seed):
    return data.build(
        "ntu_fi_humanid_reid",
        root=DATA_DIR,
        split_seed=seed,
        identities_per_batch=4,
        samples_per_identity=4,
    )


if RUN_ARCFACE:
    arcface = {}
    for encoder, kwargs in ENCODERS.items():
        arcface[encoder] = run_reid_repeats(
            lambda dm: models.build(
                encoder,
                in_shape=dm.sample_shape,
                num_classes=dm.output_dim,
                **kwargs,
            ),
            make_dm,
            seeds=SEEDS,
            max_epochs=EPOCHS,
            name=f"arcface-reid-{encoder}",
            runs_dir=RUNS_DIR,
            objective="arcface",
            detection_score="top_score",
        )

# %% [markdown]
# ## Three objectives, one benchmark
#
# All rows below threshold the top score and share seeds 42-49, so the
# comparison is paired across its full width.

# %%
TABLE_METRICS = [
    "test/rank1",
    "test/mAP",
    "test/auroc",
    "test/eer_threshold/dir",
    "test/far05_threshold/dir",
    "test/far05_threshold/far",
]


def objective_table(encoder):
    rows = {}
    for name, prefix in VARIANTS.items():
        aggregate_path = (
            REID_RUNS / f"{prefix}-{encoder}" / "aggregate_summary.json"
        )
        if not aggregate_path.exists():
            print(f"no saved results for {prefix}-{encoder}; run training first")
            continue
        stats = json.loads(aggregate_path.read_text())["metrics"]
        rows[name] = {
            metric: f"{stats[metric]['mean']:.3f} ± {stats[metric]['std']:.3f}"
            for metric in TABLE_METRICS
        }
    return pd.DataFrame(rows).T


objective_table("resnet18")

# %%
objective_table("vit")

# %%
slope_plot(per_seed["resnet18"], "resnet18, all objectives")

# %%
slope_plot(per_seed["vit"], "vit, all objectives")

# %% [markdown]
# ArcFace lands between the two, and *where* it lands depends on the
# encoder:
#
# - **ResNet18:** detection AUROC matches triplet (0.965 vs. 0.968) but the
#   per-seed view shows ArcFace inheriting triplet's failure rotations —
#   seeds 43 and 45 collapse to DIR@EER 0.00 and 0.12, exactly the
#   rotations SupCon rescues. Mean DIR@EER 0.68 vs. SupCon's 0.81, with
#   double the spread. The angular margin structures the seven *training*
#   prototypes; it does nothing to spread the unseen evaluation identities,
#   which is what the collapsed rotations need. Retrieval mAP also dips
#   (0.72 vs. 0.78-0.80).
# - **ViT:** ArcFace is the best detector of the three (AUROC 0.956 vs.
#   0.886/0.896) and on par at the EER point (DIR 0.74 vs. 0.73/0.72). For
#   the weaker encoder the margin's harder supervision appears to
#   substitute for what the metric losses were not achieving.

# %% [markdown]
# ## Cosine geometry per objective
#
# The intra- vs. inter-identity cosine histograms over one seed's test
# roles, computed from each objective's saved best checkpoint. This extends
# notebook 06's two-panel figure with the ArcFace geometry: expect ArcFace
# to compress intra-identity similarity hardest against 1.0 while leaving
# inter-identity mass high in the cone — tight clusters, small angular
# separation between unseen identities.

# %%
import torch
import torch.nn.functional as F

VIZ_SEED = SEEDS[0]
VIZ_ENCODER = "resnet18"


def load_run_net(prefix, encoder, seed):
    run_root = REID_RUNS / f"{prefix}-{encoder}" / f"seed{seed}"
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
    zs, ys = [], []
    for loader in dm.test_loaders_by_role().values():
        for x, y in loader:
            zs.append(F.normalize(net.embed(x), dim=1))
            ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


viz_dm = make_dm(VIZ_SEED)
viz_dm.setup()
embedded = {}
for objective, prefix in VARIANTS.items():
    net = load_run_net(prefix, VIZ_ENCODER, VIZ_SEED)
    if net is not None:
        embedded[objective] = embed_test_roles(net, viz_dm)

# %%
if embedded:
    fig, axes = plt.subplots(
        1, len(embedded), figsize=(5 * len(embedded), 4),
        squeeze=False, sharex=True,
    )
    bins = np.linspace(-0.2, 1.0, 61)
    for ax, (objective, (z, y)) in zip(axes[0], embedded.items()):
        similarities = z @ z.T
        same = y[:, None] == y[None, :]
        upper = np.triu(np.ones_like(same, dtype=bool), k=1)
        ax.hist(
            similarities[same & upper], bins=bins, alpha=0.6, density=True,
            label="intra-identity",
        )
        ax.hist(
            similarities[~same & upper], bins=bins, alpha=0.6, density=True,
            label="inter-identity",
        )
        ax.set(
            title=f"{objective}: cosine similarity",
            xlabel="pairwise cosine similarity",
            ylabel="density",
        )
        ax.legend(frameon=False)
    fig.tight_layout()

# %% [markdown]
# ## What would widen the known-unknown gap?
#
# The operating-point errors all live in the thin overlap between known and
# unknown top-score distributions. Three quick experiments probe what moves
# that overlap — in particular whether "more training data" helps:
#
# 1. **Fewer training identities** — train on t of the 7 training subjects
#    (`class_names` truncated before setup; the evaluation subjects of each
#    seed stay identical across conditions, so rows are paired per seed).
# 2. **Fewer samples per identity** — all 7 subjects, half the samples each.
# 3. **Probe aggregation** — no retraining: average k probe embeddings per
#    decision before scoring, from the saved SupCon checkpoints.
#
# All training conditions use the current best recipe (SupCon + top-score,
# ResNet18, 50 epochs) with 3 seeds; expect visible seed noise.

# %%
RUN_ABLATION = False
ABLATION_SEEDS = (42, 43, 44)


def make_ablation_dm(seed, train_identities=7, sample_fraction=1.0):
    dm = data.build(
        "ntu_fi_humanid_reid",
        root=DATA_DIR,
        split_seed=seed,
        identities_per_batch=4,
        samples_per_identity=4,
    )
    # Truncating class_names before setup shrinks the training set and the
    # classifier width together; evaluation roles come from the manifest and
    # are untouched.
    dm.class_names = dm.class_names[:train_identities]
    if sample_fraction < 1.0:
        original_setup = dm.setup

        def setup(stage=None, _dm=dm, _setup=original_setup):
            _setup(stage)
            by_label = {}
            for path, label in zip(_dm.train_set.files, _dm.train_set.labels):
                by_label.setdefault(label, []).append(path)
            files, labels = [], []
            for label, paths in sorted(by_label.items()):
                for path in paths[: max(2, int(len(paths) * sample_fraction))]:
                    files.append(path)
                    labels.append(label)
            _dm.train_set.files = files
            _dm.train_set.labels = labels

        dm.setup = setup
    return dm


ABLATION_CONDITIONS = {
    "ablate-train-ids-4": {"train_identities": 4},
    "ablate-train-ids-5": {"train_identities": 5},
    "ablate-train-ids-6": {"train_identities": 6},
    "ablate-train-ids-7": {"train_identities": 7},
    "ablate-half-samples-7ids": {"sample_fraction": 0.5},
}

if RUN_ABLATION:
    for condition, kwargs in ABLATION_CONDITIONS.items():
        run_reid_repeats(
            lambda dm: models.build(
                "resnet18",
                in_shape=dm.sample_shape,
                num_classes=dm.output_dim,
                base_width=32,
            ),
            lambda seed, kwargs=kwargs: make_ablation_dm(seed, **kwargs),
            seeds=ABLATION_SEEDS,
            max_epochs=50,
            name=condition,
            runs_dir=RUNS_DIR,
            objective="supcon",
            detection_score="top_score",
        )

# %%
ABLATION_METRICS = [
    "test/auroc",
    "test/eer_threshold/dir",
    "test/rank1",
    "test/mAP",
]


def ablation_table():
    rows = {}
    for condition in ABLATION_CONDITIONS:
        aggregate_path = REID_RUNS / condition / "aggregate_summary.json"
        if not aggregate_path.exists():
            print(f"no saved results for {condition}; run the ablation first")
            continue
        stats = json.loads(aggregate_path.read_text())["metrics"]
        rows[condition] = {
            metric: f"{stats[metric]['mean']:.3f} ± {stats[metric]['std']:.3f}"
            for metric in ABLATION_METRICS
        }
    return pd.DataFrame(rows).T


ablation_table()

# %% [markdown]
# In our runs, neither data axis moved detection systematically: AUROC was
# flat within noise from 4 to 7 training identities (even per-seed paired,
# the direction flips between rotations), and halving the samples cost at
# most a few points. The overlap is dominated by factors training volume
# cannot fix at this scale — whether the rotation's unknown subject happens
# to resemble an enrolled one, and the single-session, single-room nature of
# the amplitude features. Identity scaling in the metric-learning literature
# operates over orders of magnitude (hundreds to thousands of identities);
# 4 → 7 cannot show it, which is the quantitative argument for the in-house
# captures rather than for squeezing NTU-Fi harder. Note that MPS training
# is not bit-deterministic, so a re-run can jitter individual rotations.

# %%
def embed_roles_by_role(net, dm):
    roles = {}
    for role, loader in dm.test_loaders_by_role().items():
        zs, ys = [], []
        for x, y in loader:
            zs.append(F.normalize(net.embed(x), dim=1))
            ys.append(y)
        roles[role] = (torch.cat(zs), torch.cat(ys))
    return roles


def aggregate_probes(z, y, k):
    """Average consecutive same-subject embeddings in chunks of k."""
    chunks = []
    for subject in y.unique():
        zs = z[y == subject]
        for i in range(0, len(zs) - k + 1, k):
            chunks.append(F.normalize(zs[i : i + k].mean(0), dim=0))
    return torch.stack(chunks)


@torch.no_grad()
def aggregation_sweep(ks=(1, 3, 5), encoder="resnet18"):
    from sklearn.metrics import roc_auc_score

    results = {k: [] for k in ks}
    for seed in SEEDS:
        net = load_run_net(VARIANTS["supcon"], encoder, seed)
        if net is None:
            return None
        dm = make_dm(seed)
        dm.setup()
        roles = embed_roles_by_role(net, dm)
        gallery_z, gallery_y = roles["gallery"]
        for k in ks:
            known = aggregate_probes(*roles["known_probes"], k)
            unknown = aggregate_probes(*roles["unknown_probes"], k)
            probes = torch.cat([known, unknown])
            top = torch.stack(
                [
                    (probes @ gallery_z[gallery_y == c].T).amax(1)
                    for c in gallery_y.unique()
                ],
                dim=1,
            ).amax(1)
            is_known = [1.0] * len(known) + [0.0] * len(unknown)
            results[k].append(roc_auc_score(is_known, top))
    return pd.DataFrame(
        {
            f"k={k}": {
                "auroc_mean": f"{np.mean(v):.3f}",
                "auroc_std": f"{np.std(v, ddof=1):.3f}",
                "auroc_worst_rotation": f"{min(v):.3f}",
            }
            for k, v in results.items()
        }
    ).T


aggregation_sweep()

# %% [markdown]
# Probe aggregation is the one lever that reliably shrinks the overlap
# without new data or retraining: averaging k probe windows tightens both
# score distributions, lifting mean AUROC and (more importantly) the
# worst-rotation floor. In deployment this is nearly free — a person is
# present for many consecutive CSI windows — so the per-window numbers
# throughout these notebooks are conservative.

# %% [markdown]
# ## Conclusions and caveats
#
# - SupCon's aggregate DIR gain is concentrated in the previously collapsed
#   rotations; healthy rotations barely move and one rotation regresses.
#   With eight rotations, single-seed comparisons on this dataset are
#   meaningless — pair by seed, always.
# - `top_gap` is anti-correlated with being known on this protocol and
#   should not be used with three-identity galleries; the artifacts under
#   `supcon-gap-reid-*` are top-score runs that superseded it (the
#   directory name is historical).
# - ArcFace is not the missing piece on this benchmark: it detects as well
#   as triplet (and better than anything for the ViT's AUROC) but inherits
#   the triplet-style rotation collapses that SupCon fixes, because its
#   margin shapes training prototypes rather than unseen identities. Its
#   tight-cluster geometry is also exactly the regime where gap-style
#   scoring fails hardest.
# - Within NTU-Fi's range, more training data does not shrink the
#   known-unknown overlap: 4 vs. 7 training identities and full vs. half
#   samples all land within seed noise. Probe aggregation at inference is
#   the one free lever that does. Identity scaling only shows over orders
#   of magnitude, which NTU-Fi cannot provide.
# - All of this inherits NTU-Fi's limits: 14 subjects, one unknown subject
#   per rotation, single room and day. The variance across rotations — not
#   the objective choice — remains the dominant effect, which is the
#   argument for more identities (in-house captures) over more objectives.
