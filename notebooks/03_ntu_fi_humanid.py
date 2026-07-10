# %% [markdown]
# # NTU-Fi HumanID benchmark
#
# Trains the rfsensing model zoo on NTU-Fi HumanID (14 identities, samples
# of shape (3, 114, 500)) and reports test accuracy plus rank-1/rank-5
# identification accuracy (the project's person-identification metrics).
# Validation runs on the test set (no val split; SenseFi protocol).

# %%
from pathlib import Path

import pandas as pd
import torch

from rfsensing import data, models, train
from rfsensing.eval.metrics import rank_k_accuracy
from rfsensing.train import ClassificationModule

DATA_DIR = Path.cwd().resolve().parents[2] / "data"
EPOCHS = 30

dm = data.build("ntu_fi_humanid", root=DATA_DIR, batch_size=32)
print(f"{dm.name}: {dm.sample_shape}, {dm.num_classes} classes")

# %%
MODEL_CASES = {
    "mlp": {},
    "lenet": {},
    "lstm": {"seq_axis": -1},
    "resnet18": {},
    "vit": {"patch_size": 10},
}


def rank_metrics(checkpoint_path, net, dm):
    """Rank-1/rank-5 on the test set using the best checkpoint."""
    module = ClassificationModule.load_from_checkpoint(
        checkpoint_path, net=net, map_location="cpu"
    )
    module.eval()
    logits, targets = [], []
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            logits.append(module(x))
            targets.append(y)
    logits, targets = torch.cat(logits), torch.cat(targets)
    return {
        "rank1": rank_k_accuracy(logits, targets, k=1),
        "rank5": rank_k_accuracy(logits, targets, k=5),
    }


results = {}
runs = {}
nets = {}
for model_name, kwargs in MODEL_CASES.items():
    net = models.build(
        model_name, in_shape=dm.sample_shape, num_classes=dm.num_classes, **kwargs
    )
    res = train.run(net, dm, max_epochs=EPOCHS, name=model_name)
    results[model_name] = res.metrics | rank_metrics(res.checkpoint_path, net, dm)
    runs[model_name] = res
    nets[model_name] = net
    print(f"{model_name}: {results[model_name]}")

# %%
pd.DataFrame(results).T.sort_values("rank1", ascending=False)

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
