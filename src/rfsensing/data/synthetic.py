"""Synthetic classification data for tests and demos."""

import torch
from torch.utils.data import TensorDataset

from rfsensing.data import register
from rfsensing.data.base import CSIDataModule


@register("synthetic")
class SyntheticDataModule(CSIDataModule):
    """Gaussian blobs with class-dependent means — trivially learnable."""

    name = "synthetic"

    def __init__(
        self,
        sample_shape=(3, 16, 32),
        num_classes=4,
        samples_per_class=16,
        batch_size=8,
        num_workers=0,
        seed=0,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.sample_shape = tuple(sample_shape)
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.samples_per_class = samples_per_class
        self.seed = seed

    def _make_split(self, generator: torch.Generator) -> TensorDataset:
        n = self.samples_per_class * self.num_classes
        y = torch.arange(self.num_classes).repeat(self.samples_per_class)
        x = torch.randn(n, *self.sample_shape, generator=generator)
        x += y.view(-1, *([1] * len(self.sample_shape))).float() * 0.5
        return TensorDataset(x, y)

    def setup(self, stage: str | None = None) -> None:
        generator = torch.Generator().manual_seed(self.seed)
        self.train_set = self._make_split(generator)
        self.val_set = self._make_split(generator)
        self.test_set = self._make_split(generator)

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)