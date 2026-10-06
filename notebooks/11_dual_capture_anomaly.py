# %% [markdown]
# # Dual capture: empty-room anomaly detection per sniffer (USRP, ESP32-C6, Nexmon)
#
# Learns what the **empty** room looks like to each sniffer and flags every
# window that does not fit, without ever seeing a person during training
# (novelty detection). Any other condition in the manifest (`pos1`, ...) is
# the anomaly. Samples are the same 3 s amplitude windows as
# `10_dual_capture_classify.py`: (1, subcarriers, 300 steps at 100 Hz).
#
# **Protocol.** Only `empty` trials are split 60/20/20 by trial: train fits
# the detectors, val (unseen empty trials) sets each detector's threshold at
# a `TARGET_FPR` false-alarm rate, test holds the remaining empty trials plus
# *every* anomalous trial. AUROC is threshold-free; the detection rate and
# false-alarm rate at the val threshold show what a deployed detector would do.
#
# **Caveats.** With few empty trials, val/test hold one or two recordings
# each, so a single unusual empty trial moves the numbers a lot. Slow drift
# across the session (temperature, people outside the room, AGC) looks like
# an anomaly to every detector here; `09_dual_capture_stats.py` shows how
# large it is per sniffer. Interleave empty and pos recordings in time so
# drift is not confused with presence.
#
# Data layout: see `09_dual_capture_stats.py`.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch import nn
from torch.nn import functional as F

from rfsensing import data
from rfsensing.data.dual_capture import available_devices, load_trials

DATA_DIR = Path(os.environ.get("RFSENSING_DATA", Path.cwd().resolve().parents[2] / "data"))
SUBDIR = "dual"
NORMAL_LABEL = "empty"
LABELS = None  # None: every label in the manifest; e.g. ["empty", "pos1"] to restrict
WINDOW_STEPS = 300
NORMALIZATION = "train"  # empty-room train statistics; "sample" removes per-window gain
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # of the empty trials; needs >= 3 of them
SPLIT_SEED = 42
TARGET_FPR = 0.05  # threshold = this upper quantile of the val (empty) scores
AE_EPOCHS = 200
AE_PATIENCE = 20
DEVICE_NAMES = {"usrp": "USRP B200", "esp": "ESP32-C6", "nexmon": "Nexmon RPi 5"}
DEVICES = available_devices(load_trials(DATA_DIR, SUBDIR))
ND = len(DEVICES)
torch.manual_seed(SPLIT_SEED)

# %%
dms = {}
for device in DEVICES:
    dm = data.build(
        "dual_capture_anomaly",
        root=DATA_DIR,
        subdir=SUBDIR,
        device=device,
        normal_label=NORMAL_LABEL,
        labels=LABELS,
        window_steps=WINDOW_STEPS,
        normalization=NORMALIZATION,
        split_ratios=SPLIT_RATIOS,
        split_seed=SPLIT_SEED,
    )
    dm.setup()
    dms[device] = dm
    print(
        f"{DEVICE_NAMES[device]}: {dm.sample_shape}, anomalies {dm.anomaly_labels}, windows",
        {split: len(getattr(dm, f"{split}_set")) for split in ("train", "val", "test")},
    )

dm0 = dms[DEVICES[0]]
anomaly_labels = dm0.anomaly_labels
# All sniffers recorded the same trials, so the same seed gives the same split.
splits = {s: {t.name for t in ts} for s, ts in dm0.split_trials.items()}
for device in DEVICES[1:]:
    assert splits == {s: {t.name for t in ts} for s, ts in dms[device].split_trials.items()}
assert splits["train"].isdisjoint(splits["test"]) and splits["val"].isdisjoint(splits["test"])
pd.DataFrame(
    [(s, t.label, t.name) for s, ts in dm0.split_trials.items() for t in ts],
    columns=["split", "label", "trial"],
).groupby(["split", "label"]).size().unstack(fill_value=0)

# %% [markdown]
# ## Detectors
#
# **Classical**, on per-window statistics: the mean and std over time of each
# subcarrier (2 x subcarriers features). The std carries motion, the mean
# carries the static multipath a body changes. Each detector is fitted on the
# train windows only, after standardizing with their statistics; scores are
# oriented so higher means more anomalous.
#
# - `mahalanobis`: distance to a Gaussian fit (Ledoit-Wolf shrinkage, since
#   there are few windows per feature).
# - `pca_recon`: reconstruction error outside the principal subspace that
#   holds 95 % of the empty-room variance.
# - `isolation_forest`, `one_class_svm`, `lof`: the standard sklearn novelty
#   detectors.


# %%
def window_stats(dataset):
    x = dataset.tensors[0][:, 0]  # [N, S, W]
    return torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1).numpy()


class PCARecon:
    def __init__(self, variance=0.95):
        self.pca = PCA(n_components=variance)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def score(self, X):
        return ((X - self.pca.inverse_transform(self.pca.transform(X))) ** 2).mean(axis=1)


def negated(make):
    """sklearn novelty detector -> fit(X) -> score(Z), higher = more anomalous."""

    def fit(X):
        estimator = make().fit(X)
        return lambda Z: -estimator.score_samples(Z)

    return fit


def classical_detectors(n_train):
    return {
        "mahalanobis": lambda X: LedoitWolf().fit(X).mahalanobis,
        "pca_recon": lambda X: PCARecon().fit(X).score,
        "isolation_forest": negated(lambda: IsolationForest(n_estimators=300, random_state=SPLIT_SEED)),
        "one_class_svm": negated(lambda: OneClassSVM(nu=TARGET_FPR, gamma="scale")),
        "lof": negated(lambda: LocalOutlierFactor(n_neighbors=min(20, n_train - 1), novelty=True)),
    }


scores = {}  # (device, method) -> {"val": [N_val], "test": [N_test]}
for device, dm in dms.items():
    scaler = StandardScaler().fit(window_stats(dm.train_set))
    X = {s: scaler.transform(window_stats(getattr(dm, f"{s}_set"))) for s in ("train", "val", "test")}
    for method, fit in classical_detectors(len(X["train"])).items():
        score = fit(X["train"])
        scores[(device, method)] = {s: np.asarray(score(X[s])) for s in ("val", "test")}

# %% [markdown]
# **Deep**: a small 1-D convolutional autoencoder over time, subcarriers as
# channels, trained to reconstruct empty-room windows (MSE). The anomaly
# score is a window's reconstruction error. Early stopping on the val
# (empty) reconstruction loss; the anomalies are never used for selection.


# %%
class ConvAutoencoder(nn.Module):
    def __init__(self, n_sub, hidden=32, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_sub, hidden, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, latent, 5, stride=2, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent, hidden, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden, n_sub, 4, stride=2, padding=1),
        )

    def forward(self, x):  # x: [B, 1, S, W]
        z = x[:, 0]
        out = self.decoder(self.encoder(z))
        return F.interpolate(out, size=z.shape[-1], mode="linear")[:, None]


def recon_error(net, x):
    net.eval()
    with torch.no_grad():
        return ((net(x) - x) ** 2).mean(dim=(1, 2, 3)).numpy()


ae_curves = {}
for device, dm in dms.items():
    net = ConvAutoencoder(dm.sample_shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    x_val = dm.val_set.tensors[0]
    best, best_state, wait, curve = np.inf, None, 0, []
    for epoch in range(AE_EPOCHS):
        net.train()
        for x, _ in dm.train_dataloader():
            loss = F.mse_loss(net(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_loss = float(recon_error(net, x_val).mean())
        curve.append(val_loss)
        if val_loss < best:
            best, best_state, wait = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        elif (wait := wait + 1) >= AE_PATIENCE:
            break
    net.load_state_dict(best_state)
    ae_curves[device] = curve
    scores[(device, "autoencoder")] = {
        s: recon_error(net, getattr(dm, f"{s}_set").tensors[0]) for s in ("val", "test")
    }
    print(f"{DEVICE_NAMES[device]}: best val MSE {best:.3f} after {len(curve)} epochs")

METHODS = list(dict.fromkeys(m for _, m in scores))

# %% [markdown]
# ## Results
#
# Per sniffer and detector: AUROC of held-out empty vs each anomaly label and
# vs all of them, then the detection rate (TPR) and false-alarm rate (FPR) on
# test at the threshold calibrated on val. An FPR far above `TARGET_FPR`
# means the held-out empty trials differ from the training ones (drift or a
# too-small val set), not that the detector found people.


# %%
def evaluate(device, method):
    dm, s = dms[device], scores[(device, method)]
    normal = dm.test_labels == NORMAL_LABEL
    threshold = np.quantile(s["val"], 1 - TARGET_FPR)
    flagged = s["test"] > threshold
    row = {"auroc": roc_auc_score(~normal, s["test"])}
    for label in anomaly_labels:
        mask = normal | (dm.test_labels == label)
        row[f"auroc/{label}"] = roc_auc_score(dm.test_labels[mask] == label, s["test"][mask])
    row["tpr"] = flagged[~normal].mean()
    row["fpr"] = flagged[normal].mean()
    for label in anomaly_labels:
        row[f"tpr/{label}"] = flagged[dm.test_labels == label].mean()
    return row, threshold


results, thresholds = {}, {}
for key in scores:
    results[key], thresholds[key] = evaluate(*key)
table = pd.DataFrame(results).T.loc[[(d, m) for d in DEVICES for m in METHODS]]
table.index.names = ["device", "method"]
table.round(3)

# %%
auroc = table["auroc"].unstack(0)[DEVICES].loc[METHODS]
fig, ax = plt.subplots(figsize=(1.6 * ND + 3, 3.5))
image = ax.imshow(auroc.to_numpy(), vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
for (i, j), v in np.ndenumerate(auroc.to_numpy()):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.8 else "k")
ax.set(
    xticks=range(ND),
    xticklabels=[DEVICE_NAMES[d] for d in DEVICES],
    yticks=range(len(METHODS)),
    yticklabels=METHODS,
    title=f"Test AUROC: {NORMAL_LABEL} vs {', '.join(anomaly_labels)}",
)
fig.colorbar(image, ax=ax)
fig.tight_layout()

# %% [markdown]
# ### Best detector per sniffer
#
# Picked by test AUROC, so this is an optimistic upper bound with this few
# trials. ROC curves, then the score of every test window in recording order
# with the val threshold: a whole trial above the line is a detected
# presence, a single spike is a transient.

# %%
best = {d: table.loc[d, "auroc"].idxmax() for d in DEVICES}
fig, axes = plt.subplots(1, ND, figsize=(4.5 * ND, 4), squeeze=False)
for ax, device in zip(axes[0], DEVICES):
    dm, s = dms[device], scores[(device, best[device])]
    normal = dm.test_labels == NORMAL_LABEL
    for label in anomaly_labels:
        mask = normal | (dm.test_labels == label)
        fpr, tpr, _ = roc_curve(dm.test_labels[mask] == label, s["test"][mask])
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.axvline(results[(device, best[device])]["fpr"], color="grey", ls="--", lw=1)
    ax.set(title=f"{DEVICE_NAMES[device]}: {best[device]}", xlabel="FPR", ylabel="TPR")
    ax.legend()
fig.tight_layout()

# %%
colors = {label: f"C{i}" for i, label in enumerate([NORMAL_LABEL, *anomaly_labels])}
fig, axes = plt.subplots(ND, 1, figsize=(12, 3 * ND), squeeze=False)
for ax, device in zip(axes[:, 0], DEVICES):
    dm, s = dms[device], scores[(device, best[device])]
    # test trials in recording order (trial names start with the timestamp)
    order = np.argsort(dm.test_trial_names, kind="stable")
    x = np.arange(len(order))
    ax.scatter(x, s["test"][order], c=[colors[lab] for lab in dm.test_labels[order]], s=14)
    ax.axhline(thresholds[(device, best[device])], color="k", ls="--", lw=1, label="val threshold")
    names = dm.test_trial_names[order]
    for i in np.flatnonzero(names[1:] != names[:-1]) + 0.5:
        ax.axvline(i, color="grey", lw=0.5)
    ax.set(title=f"{DEVICE_NAMES[device]}: {best[device]}", ylabel="Anomaly score", yscale="log")
    ax.legend(
        handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=lab) for lab, c in colors.items()]
        + [plt.Line2D([], [], color="k", ls="--", label="val threshold")],
        loc="upper left",
    )
axes[-1, 0].set_xlabel("Test window (recording order; grey lines separate trials)")
fig.tight_layout()

# %%
fig, axes = plt.subplots(1, ND, figsize=(4.5 * ND, 3.5), squeeze=False)
for ax, device in zip(axes[0], DEVICES):
    dm, s = dms[device], scores[(device, best[device])]
    bins = np.histogram_bin_edges(np.log10(np.concatenate([s["val"], s["test"]])), 30)
    ax.hist(np.log10(s["val"]), bins, alpha=0.5, color="grey", label=f"{NORMAL_LABEL} (val)")
    for label, c in colors.items():
        ax.hist(np.log10(s["test"][dm.test_labels == label]), bins, alpha=0.5, color=c, label=f"{label} (test)")
    ax.axvline(np.log10(thresholds[(device, best[device])]), color="k", ls="--", lw=1)
    ax.set(title=f"{DEVICE_NAMES[device]}: {best[device]}", xlabel="log10 anomaly score", ylabel="Windows")
    ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ### Trial-level decision
#
# A deployed detector would vote over a recording rather than trust one
# window: the fraction of each test trial's windows above the val threshold.

# %%
trial_rows = []
for device in DEVICES:
    dm, s = dms[device], scores[(device, best[device])]
    flagged = s["test"] > thresholds[(device, best[device])]
    for name in dict.fromkeys(sorted(dm.test_trial_names)):
        mine = dm.test_trial_names == name
        trial_rows.append((DEVICE_NAMES[device], name, dm.test_labels[mine][0], flagged[mine].mean()))
pd.DataFrame(trial_rows, columns=["sniffer", "trial", "label", "flagged"]).pivot_table(
    index=["label", "trial"], columns="sniffer", values="flagged"
).round(2)

# %%
fig, ax = plt.subplots(figsize=(6, 3))
for device, curve in ae_curves.items():
    ax.plot(curve, label=DEVICE_NAMES[device])
ax.set(title="Autoencoder val (empty) reconstruction MSE", xlabel="Epoch", ylabel="MSE", yscale="log")
ax.legend()
fig.tight_layout()

# %%
summary = [
    {
        "sniffer": DEVICE_NAMES[d],
        "best detector": best[d],
        **{k: results[(d, best[d])][k] for k in ("auroc", "tpr", "fpr")},
        "mean auroc (all detectors)": table.loc[d, "auroc"].mean(),
    }
    for d in DEVICES
]
pd.DataFrame(summary).round(3)
