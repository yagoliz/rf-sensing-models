# %% [markdown]
# # Widar (BVP) benchmark
#
# Trains the rfsensing model zoo on Widar BVP gestures (22 classes, samples
# of shape (22, 20, 20): 22 time steps of 20x20 velocity maps). Validation
# runs on the test set (no val split; SenseFi protocol).

# %%
from pathlib import Path

import pandas as pd

from rfsensing import data, models, train

DATA_DIR = Path.cwd().resolve().parents[2] / "data"
EPOCHS = 30

dm = data.build("widar", root=DATA_DIR, batch_size=64)
print(f"{dm.name}: {dm.sample_shape}, {dm.num_classes} classes")

# %%
MODEL_CASES = {
    "mlp": {},
    "lenet": {},
    "lstm": {"seq_axis": 0},
    "resnet18": {},
    "vit": {"patch_size": 4},
}

results = {}
for model_name, kwargs in MODEL_CASES.items():
    net = models.build(
        model_name, in_shape=dm.sample_shape, num_classes=dm.num_classes, **kwargs
    )
    res = train.run(net, dm, max_epochs=EPOCHS, name=model_name)
    results[model_name] = res.metrics
    print(f"{model_name}: {res.metrics}")

# %%
pd.DataFrame(results).T.sort_values("test/acc", ascending=False)