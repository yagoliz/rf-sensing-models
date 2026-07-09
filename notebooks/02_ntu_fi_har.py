# %% [markdown]
# # NTU-Fi HAR benchmark
#
# Trains the rfsensing model zoo on NTU-Fi HAR (6 activities, samples of
# shape (3, 114, 500)). Note: no val split exists — validation runs on the
# test set, matching SenseFi's protocol.

# %%
from pathlib import Path

import pandas as pd

from rfsensing import data, models, train

DATA_DIR = Path.cwd().resolve().parents[2] / "data"
EPOCHS = 30

dm = data.build("ntu_fi_har", root=DATA_DIR, batch_size=32)
print(f"{dm.name}: {dm.sample_shape}, {dm.num_classes} classes")

# %%
MODEL_CASES = {
    "mlp": {},
    "lenet": {},
    "lstm": {"seq_axis": -1},
    "resnet18": {},
    "vit": {"patch_size": 10},
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