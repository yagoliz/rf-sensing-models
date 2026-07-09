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
for model_name, kwargs in MODEL_CASES.items():
    net = models.build(
        model_name, in_shape=dm.sample_shape, num_classes=dm.num_classes, **kwargs
    )
    res = train.run(net, dm, max_epochs=EPOCHS, name=model_name)
    results[model_name] = res.metrics | rank_metrics(res.checkpoint_path, net, dm)
    print(f"{model_name}: {results[model_name]}")

# %%
pd.DataFrame(results).T.sort_values("rank1", ascending=False)