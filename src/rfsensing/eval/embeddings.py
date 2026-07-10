"""Embedding-space extraction and 2-D projection plots."""

import numpy as np
import torch
import torch.nn as nn

# 8-color CVD-validated categorical palette. Beyond 8 classes we fall back
# to matplotlib's tab20 and rely on the legend plus shape-coded errors: no
# larger palette can keep every pair distinct under color-vision deficiency.
_PALETTE = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]


def _colors(n: int) -> list:
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


@torch.no_grad()
def extract(net: nn.Module, dataloader, device=None):
    """Collect ``net.embed`` features over ``dataloader``.

    Returns ``(Z, y_true, y_pred)`` numpy arrays with shapes
    ``(N, D)``, ``(N,)``, ``(N,)``.
    """
    device = torch.device(device) if device else next(net.parameters()).device
    net = net.to(device)
    was_training = net.training
    net.eval()
    zs, ys, preds = [], [], []
    for x, y in dataloader:
        z = net.embed(x.to(device))
        zs.append(z.cpu())
        ys.append(y)
        preds.append(net.head(z).argmax(dim=1).cpu())
    if was_training:
        net.train()
    return torch.cat(zs).numpy(), torch.cat(ys).numpy(), torch.cat(preds).numpy()


def project(z: np.ndarray, method: str = "pca", seed: int = 42) -> np.ndarray:
    """Project ``(N, D)`` embeddings to ``(N, 2)`` with PCA or t-SNE."""
    from sklearn.decomposition import PCA

    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(z)
    if method == "tsne":
        from sklearn.manifold import TSNE

        if z.shape[1] > 50:  # standard pre-reduction; stabilizes t-SNE
            n_components = min(50, len(z) - 1)
            z = PCA(n_components=n_components, random_state=seed).fit_transform(z)
        perplexity = min(30.0, (len(z) - 1) / 3)
        return TSNE(
            n_components=2, random_state=seed, init="pca", perplexity=perplexity
        ).fit_transform(z)
    raise ValueError(f"unknown method {method!r}; use 'pca' or 'tsne'")


def plot(z2d, y_true, class_names, y_pred=None, ax=None, title=None):
    """Scatter a 2-D projection colored by true class.

    With ``y_pred``, misclassified points are drawn as ``x`` in their true
    class's color — errors are encoded by shape, not color alone.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    colors = _colors(len(class_names))
    wrong = None if y_pred is None else np.asarray(y_pred) != np.asarray(y_true)
    for idx, name in enumerate(class_names):
        mask = np.asarray(y_true) == idx
        ok = mask if wrong is None else mask & ~wrong
        ax.scatter(
            z2d[ok, 0], z2d[ok, 1],
            s=14, color=colors[idx], alpha=0.8, linewidths=0, label=name,
        )
        if wrong is not None and (mask & wrong).any():
            bad = mask & wrong
            ax.scatter(
                z2d[bad, 0], z2d[bad, 1],
                s=26, color=colors[idx], marker="x", linewidths=1.2,
            )
    ax.set_xticks([])  # projection axes are unitless
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, markerscale=1.2, loc="best", frameon=False)
    return ax
