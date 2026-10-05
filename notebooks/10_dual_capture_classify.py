# %% [markdown]
# # Dual capture: position classification per sniffer (USRP, ESP32-C6, Nexmon)
#
# Trains the same models on each sniffer's captures from
# `csi-capturing/scripts/dual_capture.py` and compares how well they tell the
# room conditions apart (`empty`, `pos1`, `pos2`, ...). Samples are 3 s
# amplitude windows of shape (1, 30, 300): one antenna link, 30 subcarriers,
# 300 steps at 100 Hz (WiMANS' window length and subcarrier count).
#
# **Protocol.** Trials, not windows, are split 60/20/20 per label, so the test
# score measures new recordings rather than neighbouring windows of a seen
# one. Training windows overlap by half; validation and test windows do not.
# With ~10 trials per label the test set holds only ~2 trials per label, so
# expect the score to move by several points between `SPLIT_SEED`s.
#
# Data layout: see `09_dual_capture_stats.py`.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rfsensing import data, models, train
from rfsensing.data.dual_capture import available_devices, load_trials
from rfsensing.eval.metrics import confusion_matrix

DATA_DIR = Path(os.environ.get("RFSENSING_DATA", Path.cwd().resolve().parents[2] / "data"))
SUBDIR = "dual"
WINDOW_STEPS = 300
NORMALIZATION = "train"  # "sample" removes per-window gain shifts (try it for the ESP32)
SPLIT_SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
# LeNet's adaptive pooling (30 subcarriers -> 7 rows -> 4) is not implemented on
# Apple MPS; these models are small enough to train on the CPU there.
ACCELERATOR = "cpu" if torch.backends.mps.is_available() else "auto"
DEVICE_NAMES = {"usrp": "USRP B200", "esp": "ESP32-C6", "nexmon": "Nexmon RPi 5"}
DEVICES = available_devices(load_trials(DATA_DIR, SUBDIR))
ND = len(DEVICES)

# %%
dms = {}
for device in DEVICES:
    dm = data.build(
        "dual_capture",
        root=DATA_DIR,
        subdir=SUBDIR,
        device=device,
        window_steps=WINDOW_STEPS,
        normalization=NORMALIZATION,
        split_seed=SPLIT_SEED,
        batch_size=BATCH_SIZE,
    )
    dm.setup()
    dms[device] = dm
    print(
        f"{DEVICE_NAMES[device]}: {dm.sample_shape}, classes {dm.class_names}, windows",
        {split: len(getattr(dm, f"{split}_set")) for split in ("train", "val", "test")},
    )

class_names = dms[DEVICES[0]].class_names
# All sniffers recorded the same trials, so the same seed gives the same split.
splits = {s: {t.name for t in ts} for s, ts in dms[DEVICES[0]].split_trials.items()}
for device in DEVICES[1:]:
    assert splits == {s: {t.name for t in ts} for s, ts in dms[device].split_trials.items()}
assert splits["train"].isdisjoint(splits["test"]) and splits["val"].isdisjoint(splits["test"])
pd.DataFrame(
    [(s, t.label, t.name) for s, ts in dms[DEVICES[0]].split_trials.items() for t in ts],
    columns=["split", "label", "trial"],
).groupby(["split", "label"]).size().unstack()

# %%
fig, axes = plt.subplots(1, ND, figsize=(7 * ND, 3.5))
for ax, device in zip(axes, DEVICES):
    x, y = dms[device].train_set[0]
    image = ax.imshow(x[0].numpy(), aspect="auto", origin="lower")
    ax.set(title=f"{DEVICE_NAMES[device]}: {class_names[int(y)]}", xlabel="Time step", ylabel="Subcarrier")
    fig.colorbar(image, ax=ax)
fig.tight_layout()

# %% [markdown]
# ## Baseline: classical models on window statistics
#
# Per window, the mean and std over time of each subcarrier (60 features).
# A deep model that does not beat this baseline is not using the time
# structure.


# %%
def window_stats(dataset):
    x, y = dataset.tensors
    x = x[:, 0]  # [N, S, W]
    return torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1).numpy(), y.numpy()


baseline = {}
for device, dm in dms.items():
    X_train, y_train = window_stats(dm.train_set)
    X_test, y_test = window_stats(dm.test_set)
    for name, clf in {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SPLIT_SEED),
    }.items():
        clf.fit(X_train, y_train)
        baseline[(device, name)] = {"test/acc": clf.score(X_test, y_test)}
pd.DataFrame(baseline).T.unstack(0).round(3)

# %% [markdown]
# ## Deep models
#
# The same configurations as the WiMANS benchmark (`05_wimans_counting.py`),
# trained once per sniffer. Model selection uses `val/acc`; the test split is
# only read after training.

# %%
MODEL_CASES = {
    "lenet": {},
    "lstm": {"seq_axis": 2, "hidden_size": 64},
    "resnet18": {"base_width": 32},
    "vit": {"patch_size": (5, 10), "embed_dim": 64, "depth": 2},
}

results, runs, nets = {}, {}, {}
for device, dm in dms.items():
    for model_name, kwargs in MODEL_CASES.items():
        net = models.build(model_name, in_shape=dm.sample_shape, num_classes=dm.num_classes, **kwargs)
        result = train.run(
            net, dm, max_epochs=EPOCHS, seed=SPLIT_SEED, accelerator=ACCELERATOR, name=f"{device}-{model_name}"
        )
        results[(device, model_name)] = result.metrics | {"val/acc (best)": result.best_score}
        runs[(device, model_name)] = result
        nets[(device, model_name)] = net
        print(device, model_name, result.metrics)

table = pd.DataFrame(results).T
table.index.names = ["device", "model"]
table[["val/acc (best)", "test/acc"]].unstack(0).round(3)

# %% [markdown]
# ## Best model per sniffer
#
# Chosen by validation accuracy. Rows are the true condition, columns the
# prediction, normalized per row.


# %%
def predict(net, loader):
    device = next(net.parameters()).device
    net.eval()
    logits, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(net(x.to(device)).cpu())
            targets.append(y)
    return torch.cat(logits), torch.cat(targets)


fig, axes = plt.subplots(1, ND, figsize=(6 * ND, 5))
for ax, device in zip(axes, DEVICES):
    best = max(MODEL_CASES, key=lambda m: runs[(device, m)].best_score)
    net = train.load_best_net(nets[(device, best)], runs[(device, best)])
    logits, targets = predict(net, dms[device].test_dataloader())
    matrix = confusion_matrix(logits, targets, num_classes=len(class_names)).float()
    matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1)
    ax.imshow(matrix.numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    acc = (logits.argmax(dim=1) == targets).float().mean().item()
    ax.set(
        title=f"{DEVICE_NAMES[device]}: {best} (test acc {acc:.2f})",
        xlabel="Predicted",
        ylabel="True",
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
fig.tight_layout()

# %% [markdown]
# ## Summary
#
# One row per sniffer: the best classical baseline next to the best deep
# model (by validation accuracy) and its test accuracy.

# %%
summary = []
for device in DEVICES:
    best = max(MODEL_CASES, key=lambda m: runs[(device, m)].best_score)
    summary.append(
        {
            "sniffer": DEVICE_NAMES[device],
            "best baseline test/acc": max(v["test/acc"] for (d, _), v in baseline.items() if d == device),
            "best deep model": best,
            "deep test/acc": results[(device, best)]["test/acc"],
            "chance": 1 / len(class_names),
        }
    )
pd.DataFrame(summary).round(3)
