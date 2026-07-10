# %% [markdown]
# # UT-HAR benchmark
#
# Trains the rfsensing model zoo on UT-HAR (7 activities, samples of shape
# (1, 250, 90)) and tabulates test accuracy, comparable to SenseFi's tables.

# %%
from pathlib import Path

import pandas as pd

from rfsensing import data, models, train

DATA_DIR = Path.cwd().resolve().parents[2] / "data"
EPOCHS = 50

dm = data.build("ut_har", root=DATA_DIR, batch_size=64)
print(f"{dm.name}: {dm.sample_shape}, {dm.num_classes} classes")

# %%
MODEL_CASES = {
    "mlp": {},
    "lenet": {},
    "lstm": {"seq_axis": 1},
    "resnet18": {},
    "vit": {"patch_size": 10},
}

results = {}
runs = {}
nets = {}
for model_name, kwargs in MODEL_CASES.items():
    net = models.build(
        model_name, in_shape=dm.sample_shape, num_classes=dm.num_classes, **kwargs
    )
    res = train.run(net, dm, max_epochs=EPOCHS, name=model_name)
    results[model_name] = res.metrics
    runs[model_name] = res
    nets[model_name] = net
    print(f"{model_name}: {res.metrics}")

# %%
pd.DataFrame(results).T.sort_values("test/acc", ascending=False)

# %% [markdown]
# ## Embedding space
#
# t-SNE projection of each model's pre-classifier features on the test set,
# from the best (val/acc) checkpoint. Misclassified samples are drawn as ×
# in their true class's color.

# %%
import matplotlib.pyplot as plt

from rfsensing.eval import embeddings

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (model_name, net) in zip(axes.flat, nets.items()):
    train.load_best_net(net, runs[model_name])
    z, y_true, y_pred = embeddings.extract(net, dm.test_dataloader(), device="cpu")
    z2d = embeddings.project(z, method="tsne")
    embeddings.plot(z2d, y_true, dm.class_names, y_pred=y_pred, ax=ax, title=model_name)
for ax in axes.flat[len(nets):]:
    ax.axis("off")
fig.tight_layout()
