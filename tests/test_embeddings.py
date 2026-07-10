import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from rfsensing import data, models
from rfsensing.eval import embeddings


def _fixtures():
    dm = data.build("synthetic", batch_size=8)
    dm.setup("test")
    net = models.build(
        "mlp", in_shape=dm.sample_shape, num_classes=dm.num_classes, hidden_dims=(16,)
    )
    return dm, net


def test_extract_shapes_and_dtypes():
    dm, net = _fixtures()
    z, y_true, y_pred = embeddings.extract(net, dm.test_dataloader())
    n = len(dm.test_set)
    assert z.shape == (n, 16) and z.dtype == np.float32
    assert y_true.shape == (n,) and y_pred.shape == (n,)
    assert set(np.unique(y_pred)) <= set(range(dm.num_classes))


def test_extract_leaves_training_mode_intact():
    dm, net = _fixtures()
    net.train()
    embeddings.extract(net, dm.test_dataloader())
    assert net.training


def test_project_pca_and_tsne():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(64, 16)).astype(np.float32)
    for method in ("pca", "tsne"):
        z2 = embeddings.project(z, method=method)
        assert z2.shape == (64, 2)


def test_project_tsne_handles_wide_embeddings():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(40, 128)).astype(np.float32)
    assert embeddings.project(z, method="tsne").shape == (40, 2)


def test_project_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        embeddings.project(np.zeros((4, 8), dtype=np.float32), method="umap")


def test_plot_smoke():
    rng = np.random.default_rng(0)
    z2d = rng.normal(size=(32, 2))
    y = np.repeat(np.arange(4), 8)
    y_pred = y.copy()
    y_pred[0] = 3  # one misclassification exercises the error-marker path
    _, ax = plt.subplots()
    out = embeddings.plot(
        z2d, y, [f"class_{i}" for i in range(4)], y_pred=y_pred, ax=ax, title="t"
    )
    assert out is ax
    assert len(ax.collections) > 0
    plt.close("all")
