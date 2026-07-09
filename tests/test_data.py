import pytest
import torch

from rfsensing import data
from conftest import DATA_ROOT, requires_data


def test_unknown_dataset_lists_available():
    with pytest.raises(KeyError, match="Unknown dataset 'nope'"):
        data.build("nope")


def test_synthetic_metadata():
    dm = data.build("synthetic", sample_shape=(2, 8, 8), num_classes=3)
    assert dm.name == "synthetic"
    assert dm.sample_shape == (2, 8, 8)
    assert dm.num_classes == 3
    assert len(dm.class_names) == 3


def test_synthetic_batches():
    dm = data.build("synthetic", batch_size=8)
    dm.setup("fit")
    dm.setup("test")
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (8, *dm.sample_shape)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    for loader in (dm.val_dataloader(), dm.test_dataloader()):
        xb, yb = next(iter(loader))
        assert xb.shape[1:] == dm.sample_shape


def test_ut_har_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Expected layout"):
        data.build("ut_har", root=tmp_path)


@pytest.mark.data
@requires_data
def test_ut_har_real_data():
    dm = data.build("ut_har", root=DATA_ROOT, batch_size=4)
    dm.setup("fit")
    dm.setup("test")
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, 1, 250, 90)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert dm.num_classes == 7
    assert 0 <= int(y.min()) and int(y.max()) < 7