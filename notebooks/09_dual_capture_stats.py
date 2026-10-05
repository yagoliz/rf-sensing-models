# %% [markdown]
# # Dual capture: per-position statistics
#
# Compares the sniffers (USRP, ESP32-C6, and the Nexmon Pi when recorded) in
# the captures recorded by
# `csi-capturing/scripts/dual_capture.py` across the room conditions
# (`empty`, `pos1`, `pos2`, ...). All sniffers listened to the same frames
# from one router at the same time, so differences between them are the
# sniffer (radio, antenna, placement), not the scene.
#
# Data layout: copy the capture's `--output-dir` (default
# `csi-capturing/data/dual`) to `../../data/dual`, or set `RFSENSING_DATA`.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

from rfsensing.data.dual_capture import available_devices, load_trials, window_starts

DATA_DIR = Path(os.environ.get("RFSENSING_DATA", Path.cwd().resolve().parents[2] / "data"))
SUBDIR = "dual"
WINDOW_STEPS = 300  # 3 s at 100 Hz
MIN_VALID = 0.9
DEVICE_NAMES = {"usrp": "USRP B200", "esp": "ESP32-C6", "nexmon": "Nexmon RPi 5"}

trials = load_trials(DATA_DIR, SUBDIR)
DEVICES = available_devices(trials)
ND = len(DEVICES)
labels = sorted({t.label for t in trials}, key=lambda lab: (lab != "empty", lab))
colors = {label: f"C{i}" for i, label in enumerate(labels)}
print(f"{len(trials)} trials, labels {labels}, sniffers {DEVICES}, {len(trials[0].subcarriers)} subcarriers")

# %% [markdown]
# ## Capture quality
#
# Rates are frames/s from the router after filtering; `*_valid` is the
# fraction of 10 ms grid steps that had a fresh frame (the rest repeat the
# previous one). A sniffer far below 100 Hz, or a trial that stands out, is
# worth checking before reading anything into the comparisons below.

# %%
manifest = pd.DataFrame([t.manifest for t in trials])
quality_cols = ["overlap_s", "windows", *(f"{d}_rate_hz" for d in DEVICES), *(f"{d}_valid" for d in DEVICES)]
if "nexmon_vs_usrp_ms" in manifest:
    quality_cols.append("nexmon_vs_usrp_ms")  # frame-matched alignment check, ~0 is good
manifest.groupby("label")[quality_cols].agg(["mean", "min"]).round(2).loc[labels]

# %%
fig, axes = plt.subplots(1, ND, figsize=(7 * ND, 3.5), sharey=True)
for ax, device in zip(axes, DEVICES):
    ax.bar(
        range(len(manifest)),
        manifest[f"{device}_rate_hz"],
        color=[colors[label] for label in manifest["label"]],
    )
    ax.axhline(trials[0].rate, color="black", linestyle="--", linewidth=1)
    ax.set(title=DEVICE_NAMES[device], xlabel="Trial (recording order)", ylabel="Frames/s")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
axes[-1].legend(handles, labels, loc="lower right")
fig.tight_layout()

# %% [markdown]
# ## Per-window features
#
# Each trial is cut into non-overlapping windows of `WINDOW_STEPS` (the same
# windows the classifier sees). Per window:
#
# - **level**: mean amplitude in dB, the received signal strength proxy;
# - **motion**: mean over subcarriers of the temporal std of the
#   per-frame-normalized amplitude (removes AGC level shifts, keeps how much
#   the channel shape moves);
# - **profile**: mean amplitude per subcarrier divided by the window mean,
#   the shape of the frequency response, where a person's position shows up.
#
# The two sniffers use different amplitude units, so compare labels within a
# device, not levels across devices.


# %%
def window_features(trial, device):
    amp = trial.amplitude(device)
    rows, profiles = [], []
    for s in window_starts(trial.valid[device], WINDOW_STEPS, WINDOW_STEPS, MIN_VALID):
        w = amp[s : s + WINDOW_STEPS]
        norm = w / w.mean(axis=1, keepdims=True).clip(1e-9)
        profile = w.mean(axis=0)
        rows.append(
            {
                "trial": trial.name,
                "label": trial.label,
                "device": device,
                "start_s": round(float(trial.t[s] - trial.t[0]), 1),
                "time": trial.t[s],
                "level_db": 20 * np.log10(w.mean() + 1e-9),
                "motion": float(norm.std(axis=0).mean()),
            }
        )
        profiles.append(profile / profile.mean())
    return rows, profiles


feature_rows = []
profiles = {device: [] for device in DEVICES}
profile_labels = {device: [] for device in DEVICES}
for trial in trials:
    for device in DEVICES:
        rows, profs = window_features(trial, device)
        feature_rows += rows
        profiles[device] += profs
        profile_labels[device] += [trial.label] * len(profs)
features = pd.DataFrame(feature_rows)
profiles = {device: np.array(p) for device, p in profiles.items()}
profile_labels = {device: np.array(p) for device, p in profile_labels.items()}
features.groupby(["device", "label"])[["level_db", "motion"]].agg(["mean", "std", "count"]).round(3)

# %%
fig, axes = plt.subplots(2, ND, figsize=(6 * ND, 7))
for col, device in enumerate(DEVICES):
    sub = features[features["device"] == device]
    for row, (feature, unit) in enumerate([("level_db", "dB"), ("motion", "normalized std")]):
        ax = axes[row, col]
        groups = [sub.loc[sub["label"] == label, feature] for label in labels]
        box = ax.boxplot(groups, tick_labels=labels, patch_artist=True)
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.5)
        ax.set(title=f"{DEVICE_NAMES[device]}: {feature}", ylabel=unit)
fig.tight_layout()

# %% [markdown]
# ### Frequency-response profile per position
#
# Mean ± std over windows. Where the curves separate (relative to their
# spread) is where a classifier can tell the positions apart.

# %%
subcarriers = trials[0].subcarriers
fig, axes = plt.subplots(1, ND, figsize=(7 * ND, 4))
for ax, device in zip(axes, DEVICES):
    for label in labels:
        p = profiles[device][profile_labels[device] == label]
        mean, std = p.mean(axis=0), p.std(axis=0)
        ax.plot(subcarriers, mean, color=colors[label], label=label)
        ax.fill_between(subcarriers, mean - std, mean + std, color=colors[label], alpha=0.2)
    ax.set(title=DEVICE_NAMES[device], xlabel="Subcarrier index", ylabel="Amplitude / window mean")
    ax.legend()
fig.tight_layout()

# %% [markdown]
# ### Example trials
#
# Amplitude over time for the first trial of each label (rows: sniffer).
# Static positions should look like horizontal stripes; vertical structure
# is movement or gain changes.

# %%
first = {label: next(t for t in trials if t.label == label) for label in labels}
fig, axes = plt.subplots(ND, len(labels), figsize=(5 * len(labels), 3 * ND), squeeze=False)
for row, device in enumerate(DEVICES):
    for col, label in enumerate(labels):
        trial = first[label]
        amp = trial.amplitude(device)
        ax = axes[row, col]
        image = ax.imshow(
            amp.T,
            aspect="auto",
            origin="lower",
            extent=(0, trial.t[-1] - trial.t[0], subcarriers[0], subcarriers[-1]),
        )
        ax.set(title=f"{DEVICE_NAMES[device]}: {label}", xlabel="Time [s]", ylabel="Subcarrier")
        fig.colorbar(image, ax=ax)
fig.tight_layout()

# %% [markdown]
# ## Separability
#
# PCA of the window profiles (fitted per sniffer), colored by label, and the
# one-way ANOVA F statistic of each subcarrier across labels. Well-separated
# clusters and large F values mean simple features already distinguish the
# positions; overlapping clusters mean the classifier has to work harder.

# %%
fig, axes = plt.subplots(2, ND, figsize=(6.5 * ND, 9))
for col, device in enumerate(DEVICES):
    X, y = profiles[device], profile_labels[device]
    z = PCA(n_components=2).fit(X)
    z2 = z.transform(X)
    ax = axes[0, col]
    for label in labels:
        ax.scatter(*z2[y == label].T, s=10, alpha=0.6, color=colors[label], label=label)
    ratio = z.explained_variance_ratio_
    ax.set(
        title=f"{DEVICE_NAMES[device]}: profile PCA",
        xlabel=f"PC1 ({ratio[0]:.0%})",
        ylabel=f"PC2 ({ratio[1]:.0%})",
    )
    ax.legend()
    f_stat, _ = f_classif(X, y)
    axes[1, col].bar(subcarriers, f_stat)
    axes[1, col].set(
        title=f"{DEVICE_NAMES[device]}: ANOVA F per subcarrier",
        xlabel="Subcarrier index",
        ylabel="F",
    )
fig.tight_layout()

# %% [markdown]
# ## Drift across the session
#
# Trial-level mean of the features in recording order. If a label's trials
# drift over the session (temperature, furniture, AGC), a classifier can learn
# the time of recording instead of the position; interleaved labels with
# overlapping drift are fine.

# %%
trial_means = (
    features.groupby(["device", "trial", "label"])[["time", "level_db", "motion"]]
    .mean()
    .reset_index()
)
t_start = trial_means["time"].min()
fig, axes = plt.subplots(2, ND, figsize=(6.5 * ND, 7), sharex=True)
for col, device in enumerate(DEVICES):
    sub = trial_means[trial_means["device"] == device]
    for row, feature in enumerate(["level_db", "motion"]):
        ax = axes[row, col]
        for label in labels:
            s = sub[sub["label"] == label]
            ax.plot((s["time"] - t_start) / 60, s[feature], "o", color=colors[label], label=label)
        ax.set(title=f"{DEVICE_NAMES[device]}: {feature}", ylabel=feature)
    axes[1, col].set_xlabel("Minutes since first trial")
axes[0, 0].legend()
fig.tight_layout()

# %% [markdown]
# ## Cross-sniffer agreement
#
# For every pair of sniffers: correlation between their per-step mean
# amplitude (z-scored per trial), and the lag of the cross-correlation peak
# within ±0.5 s. The sniffers see different propagation paths, so a low
# correlation in a static trial is expected; when someone moves, both should
# react together and the peak lag checks the time alignment (expect a few
# grid steps at most). With the Nexmon Pi, the manifest's `nexmon_vs_usrp_ms`
# is a sharper check: the same frames matched by sequence number.

# %%
from itertools import combinations


def agreement(trial, dev_a, dev_b, max_lag=50):
    both = trial.valid[dev_a] & trial.valid[dev_b]
    a = trial.amplitude(dev_a).mean(axis=1)[both]
    b = trial.amplitude(dev_b).mean(axis=1)[both]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    lags = np.arange(-max_lag, max_lag + 1)
    xc = [np.mean(a[max(0, -k) : len(a) - max(0, k)] * b[max(0, k) : len(b) - max(0, -k)]) for k in lags]
    best = int(np.argmax(np.abs(xc)))
    return {
        "pair": f"{dev_a}-{dev_b}",
        "trial": trial.name,
        "label": trial.label,
        "corr": float(np.mean(a * b)),
        "peak_corr": float(xc[best]),
        "peak_lag_ms": float(lags[best] * 1000 / trial.rate),
    }


agree = pd.DataFrame(
    [agreement(t, a, b) for t in trials for a, b in combinations(DEVICES, 2)]
)
agree.groupby(["pair", "label"])[["corr", "peak_corr", "peak_lag_ms"]].agg(["mean", "std"]).round(3)
