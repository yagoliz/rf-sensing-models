# %% [markdown]
# # WiMANS people counting
#
# WiMANS contains 0-5 simultaneous users recorded across three rooms and two
# Wi-Fi bands. This notebook compares ordered group-size classification with
# scalar count regression using group-held-out splits. It directly targets the
# group-size MAE and ±1-person metrics in `MAIN_PROJECT.md`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from rfsensing import data, models, train
from rfsensing.eval.metrics import confusion_matrix

DATA_DIR = Path.cwd().resolve().parents[2] / "data"
TIME_STEPS = 300
NORMALIZATION = "train"
POOLING = "mean"
SPLIT_SEED = 42
BATCH_SIZE = 32
EPOCHS = 30

# %% [markdown]
# ## Dataset and protocol
#
# The loader requires all 11,286 amplitude files. It accepts both the official
# `WiMANS/wifi_csi/amp` layout and the flat `WiMANS/amp` layout. Samples with
# zero people are intentional members of the six count classes.

# %%
annotations = pd.read_csv(DATA_DIR / "WiMANS" / "annotation.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
annotations["number_of_users"].value_counts().sort_index().plot.bar(
    ax=axes[0], title="People count"
)
annotations["environment"].value_counts().plot.bar(
    ax=axes[1], title="Environment"
)
annotations["wifi_band"].value_counts().plot.bar(
    ax=axes[2], title="Wi-Fi band"
)
fig.tight_layout()

# %%
COMMON_DATA = {
    "root": DATA_DIR,
    "raw_time_steps": 3000,
    "time_steps": TIME_STEPS,
    "pad_side": "left",
    "pooling": POOLING,
    "normalization": NORMALIZATION,
    "split_strategy": "group",
    "split_ratios": (0.70, 0.15, 0.15),
    "split_seed": SPLIT_SEED,
    "batch_size": BATCH_SIZE,
}
dm_class = data.build("wimans", target="classification", **COMMON_DATA)
dm_class.setup()
print(
    dm_class.sample_shape,
    {name: len(records) for name, records in dm_class.split_records.items()},
)
train_groups = dm_class.split_group_ids["train"]
val_groups = dm_class.split_group_ids["val"]
test_groups = dm_class.split_group_ids["test"]
assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# %% [markdown]
# This split is intentionally harder than the official random 80/20 WiMANS
# benchmark: every `act_<group>_*` user combination is confined to one split,
# so the resulting scores are not directly comparable to the paper's table.

# %%
x_example, y_example = dm_class.train_set[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for link, ax in enumerate(axes):
    image = ax.imshow(x_example[link].numpy(), aspect="auto", origin="lower")
    ax.set(title=f"Antenna link {link}", xlabel="Time bin", ylabel="Subcarrier")
    fig.colorbar(image, ax=ax)
fig.suptitle(f"Processed WiMANS sample, count={int(y_example)}")
fig.tight_layout()

# %% [markdown]
# ## Model configurations
#
# The raw flattened representation makes the current MLP unnecessarily large,
# so this benchmark uses temporal and convolutional models. Change this mapping
# to adjust model capacity without changing the DataModule.

# %%
MODEL_CASES = {
    "lenet": {},
    "lstm": {"seq_axis": 2, "hidden_size": 64},
    "resnet18": {"base_width": 32},
    "vit": {"patch_size": (5, 10), "embed_dim": 64, "depth": 2},
}


def run_models(dm, task_name):
    results = {}
    runs = {}
    nets = {}
    for model_name, kwargs in MODEL_CASES.items():
        net = models.build(
            model_name,
            in_shape=dm.sample_shape,
            num_classes=dm.output_dim,
            **kwargs,
        )
        result = train.run(
            net,
            dm,
            max_epochs=EPOCHS,
            name=f"{task_name}-{model_name}",
        )
        results[model_name] = result.metrics
        runs[model_name] = result
        nets[model_name] = net
        print(model_name, result.metrics)
    table = pd.DataFrame(results).T
    table["selection/monitor"] = pd.Series(
        {name: result.monitor for name, result in runs.items()}
    )
    table["selection/value"] = pd.Series(
        {name: result.best_score for name, result in runs.items()}
    )
    return table, runs, nets


def select_best_run(runs):
    modes = {result.monitor_mode for result in runs.values()}
    if len(modes) != 1:
        raise ValueError("all compared runs must use the same monitor mode")
    scores = pd.Series(
        {name: result.best_score for name, result in runs.items()}
    )
    return scores.idxmin() if modes.pop() == "min" else scores.idxmax()


# %% [markdown]
# ## Six-class group-size classification

# %%
classification_table, classification_runs, classification_nets = run_models(
    dm_class, "count-classification"
)
classification_table[
    ["test/acc", "test/mae", "test/within_1"]
].sort_values("test/mae")

# %% [markdown]
# ## Scalar count regression
#
# Regression MAE uses the raw scalar prediction. Rounded exact accuracy is
# included only as an interpretable secondary metric.

# %%
dm_regression = data.build("wimans", target="regression", **COMMON_DATA)
regression_table, regression_runs, regression_nets = run_models(
    dm_regression, "count-regression"
)
regression_table[
    ["test/rounded_acc", "test/mae", "test/within_1"]
].sort_values("test/mae")

# %%
comparison = pd.concat(
    {
        "classification": classification_table.rename(
            columns={"test/acc": "test/exact_acc"}
        ),
        "regression": regression_table.rename(
            columns={"test/rounded_acc": "test/exact_acc"}
        ),
    },
    names=["task", "model"],
)
comparison[["test/exact_acc", "test/mae", "test/within_1"]].sort_values(
    "test/mae"
)

# %% [markdown]
# ## Best-model diagnostics
#
# Model selection uses each run's validation checkpoint monitor (`val/acc` for
# classification and `val/mae` for regression). The test split is used only
# after selection for the plots below.

# %%
def collect_predictions(net, loader, regression=False):
    device = next(net.parameters()).device
    predictions = []
    targets = []
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            output = net(x.to(device)).cpu()
            predictions.append(output.reshape(-1) if regression else output)
            targets.append(y.cpu())
    return torch.cat(predictions), torch.cat(targets)


best_class_name = select_best_run(classification_runs)
best_class_net = train.load_best_net(
    classification_nets[best_class_name],
    classification_runs[best_class_name],
)
class_logits, class_targets = collect_predictions(
    best_class_net, dm_class.test_dataloader()
)
matrix = confusion_matrix(class_logits, class_targets, num_classes=6)
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(matrix.numpy(), cmap="Blues")
ax.set(
    title=f"Count confusion matrix: {best_class_name}",
    xlabel="Predicted count",
    ylabel="True count",
    xticks=range(6),
    yticks=range(6),
)
fig.colorbar(image, ax=ax)
fig.tight_layout()

# %%
best_reg_name = select_best_run(regression_runs)
best_reg_net = train.load_best_net(
    regression_nets[best_reg_name],
    regression_runs[best_reg_name],
)
reg_predictions, reg_targets = collect_predictions(
    best_reg_net, dm_regression.test_dataloader(), regression=True
)
residuals = reg_predictions - reg_targets
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(reg_targets.numpy(), reg_predictions.numpy(), alpha=0.35)
axes[0].plot([0, 5], [0, 5], color="black", linestyle="--")
axes[0].set(
    title=f"Regression predictions: {best_reg_name}",
    xlabel="True count",
    ylabel="Raw predicted count",
)
axes[1].hist(residuals.numpy(), bins=30)
axes[1].set(title="Residual distribution", xlabel="Prediction − target")
fig.tight_layout()
