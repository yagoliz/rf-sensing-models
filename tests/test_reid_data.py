from collections import Counter

import pytest
import torch

from rfsensing import data
from rfsensing.data.reid import (
    IdentityBatchSampler,
    IdentitySplit,
    make_identity_split,
)

from tests.conftest import DATA_ROOT, requires_ntu_humanid

IDENTITIES = [f"{i:03d}" for i in range(1, 15)]


def test_split_fixed_seed_is_deterministic():
    a = make_identity_split(IDENTITIES, seed=7)
    b = make_identity_split(IDENTITIES, seed=7)
    assert a == b


def test_split_input_order_does_not_matter():
    a = make_identity_split(IDENTITIES, seed=7)
    b = make_identity_split(list(reversed(IDENTITIES)), seed=7)
    assert a == b


def test_split_different_seeds_differ():
    splits = {make_identity_split(IDENTITIES, seed=s).train for s in range(8)}
    assert len(splits) > 1


def test_split_default_counts():
    split = make_identity_split(IDENTITIES, seed=0)
    assert len(split.train) == 7
    assert len(split.val_enrolled) == 2
    assert len(split.val_unknown) == 1
    assert len(split.test_enrolled) == 3
    assert len(split.test_unknown) == 1


def test_split_assigns_every_identity_exactly_once():
    split = make_identity_split(IDENTITIES, seed=3)
    assigned = (
        split.train
        + split.val_enrolled
        + split.val_unknown
        + split.test_enrolled
        + split.test_unknown
    )
    assert sorted(assigned) == sorted(IDENTITIES)


def test_split_to_dict_round_trip():
    split = make_identity_split(IDENTITIES, seed=11)
    d = split.to_dict()
    assert d["seed"] == 11
    assert d["train"] == list(split.train)
    assert d["val_enrolled"] == list(split.val_enrolled)
    assert d["val_unknown"] == list(split.val_unknown)
    assert d["test_enrolled"] == list(split.test_enrolled)
    assert d["test_unknown"] == list(split.test_unknown)


def test_split_rejects_duplicate_identities():
    with pytest.raises(ValueError, match="duplicate"):
        make_identity_split(IDENTITIES + [IDENTITIES[0]], seed=0)


def test_split_rejects_count_mismatch():
    with pytest.raises(ValueError, match="14 identities"):
        make_identity_split(IDENTITIES, seed=0, train_count=8)
    with pytest.raises(ValueError, match="12 identities"):
        make_identity_split(IDENTITIES[:12], seed=0)


def test_split_rejects_empty_roles():
    with pytest.raises(ValueError, match="at least 1"):
        make_identity_split(
            IDENTITIES,
            seed=0,
            train_count=8,
            val_enrolled_count=2,
            val_unknown_count=0,
            test_enrolled_count=3,
            test_unknown_count=1,
        )


def test_validate_rejects_overlapping_roles():
    split = IdentitySplit(
        seed=0,
        train=("a", "b"),
        val_enrolled=("b",),
        val_unknown=("c",),
        test_enrolled=("d",),
        test_unknown=("e",),
    )
    with pytest.raises(ValueError, match="train.*val_enrolled"):
        split.validate()


def test_validate_rejects_duplicates_within_role():
    split = IdentitySplit(
        seed=0,
        train=("a", "a"),
        val_enrolled=("b",),
        val_unknown=("c",),
        test_enrolled=("d",),
        test_unknown=("e",),
    )
    with pytest.raises(ValueError, match="duplicate"):
        split.validate()


def test_validate_rejects_empty_role():
    split = IdentitySplit(
        seed=0,
        train=("a",),
        val_enrolled=(),
        val_unknown=("c",),
        test_enrolled=("d",),
        test_unknown=("e",),
    )
    with pytest.raises(ValueError, match="val_enrolled"):
        split.validate()


# --- IdentityBatchSampler ---

LABELS = [0] * 6 + [1] * 6 + [2] * 6 + [3] * 3


def _batches(sampler):
    return [list(batch) for batch in sampler]


def test_sampler_batches_have_p_identities_k_samples():
    sampler = IdentityBatchSampler(LABELS, 3, 4, seed=0)
    for batch in sampler:
        assert len(batch) == 12
        counts = Counter(LABELS[i] for i in batch)
        assert len(counts) == 3
        assert all(count == 4 for count in counts.values())


def test_sampler_epoch_length_covers_dataset():
    sampler = IdentityBatchSampler(LABELS, 3, 4, seed=0)
    # ceil(21 / 12) == 2
    assert len(sampler) == 2
    assert len(_batches(sampler)) == 2
    tiny = IdentityBatchSampler([0, 0, 1, 1], 2, 2, seed=0)
    assert len(tiny) == 1


def test_sampler_deterministic_for_seed_and_epoch():
    a = IdentityBatchSampler(LABELS, 3, 4, seed=5)
    b = IdentityBatchSampler(LABELS, 3, 4, seed=5)
    a.set_epoch(2)
    b.set_epoch(2)
    assert _batches(a) == _batches(b)


def test_sampler_changes_across_epochs():
    sampler = IdentityBatchSampler(LABELS, 3, 4, seed=5)
    sampler.set_epoch(0)
    first = _batches(sampler)
    sampler.set_epoch(1)
    second = _batches(sampler)
    assert first != second


def test_sampler_replacement_for_small_identities():
    # identity 3 has 3 samples < K=4: replacement must fill the batch.
    sampler = IdentityBatchSampler(LABELS, 4, 4, seed=1)
    for batch in sampler:
        counts = Counter(LABELS[i] for i in batch)
        assert counts[3] == 4 if 3 in counts else True
    # identities with >= K samples must not repeat indices within a batch
    sampler = IdentityBatchSampler(LABELS, 3, 4, seed=1)
    for batch in sampler:
        by_label = {}
        for i in batch:
            by_label.setdefault(LABELS[i], []).append(i)
        for label, indices in by_label.items():
            if LABELS.count(label) >= 4:
                assert len(set(indices)) == len(indices)


def test_sampler_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="identities_per_batch"):
        IdentityBatchSampler(LABELS, 1, 4)
    with pytest.raises(ValueError, match="samples_per_identity"):
        IdentityBatchSampler(LABELS, 2, 1)
    with pytest.raises(ValueError, match="distinct identities"):
        IdentityBatchSampler([0] * 8 + [1] * 8, 3, 4)
    with pytest.raises(ValueError, match="empty"):
        IdentityBatchSampler([], 2, 2)
    with pytest.raises(ValueError, match="integer"):
        IdentityBatchSampler([0.5] * 4 + [1.5] * 4, 2, 2)


# --- NTUFiHumanIDReIDDataModule ---


@pytest.fixture(scope="module")
def reid_dm(fake_ntu_root):
    dm = data.build(
        "ntu_fi_humanid_reid",
        root=fake_ntu_root,
        split_seed=42,
        identities_per_batch=3,
        samples_per_identity=2,
        eval_batch_size=4,
    )
    dm.setup("fit")
    dm.setup("test")
    return dm


def test_reid_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Expected layout"):
        data.build("ntu_fi_humanid_reid", root=tmp_path)


def test_reid_metadata(reid_dm):
    assert reid_dm.name == "ntu_fi_humanid_reid"
    assert reid_dm.task_type == "reid"
    assert reid_dm.sample_shape == (3, 114, 500)
    assert reid_dm.output_dim == 7
    assert reid_dm.num_classes == 7
    assert reid_dm.checkpoint_monitor == "val/mAP"
    assert reid_dm.checkpoint_mode == "max"
    assert len(reid_dm.identity_names) == 14
    assert reid_dm.identity_names == sorted(reid_dm.identity_names)


def test_reid_split_manifest(reid_dm):
    manifest = reid_dm.split_manifest
    assert manifest.seed == 42
    assert manifest == make_identity_split(reid_dm.identity_names, seed=42)
    assert list(reid_dm.class_names) == list(manifest.train)


def test_reid_train_batches_are_p_by_k(reid_dm):
    loader = reid_dm.train_dataloader()
    assert isinstance(loader.batch_sampler, IdentityBatchSampler)
    x, y = next(iter(loader))
    assert x.shape == (6, 3, 114, 500)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    counts = Counter(y.tolist())
    assert len(counts) == 3
    assert all(count == 2 for count in counts.values())


def test_reid_train_labels_are_contiguous(reid_dm):
    labels = set(reid_dm.train_set.labels)
    assert labels == set(range(7))


def test_reid_role_sources_follow_protocol(reid_dm):
    manifest = reid_dm.split_manifest
    for split_roles, loaders in (
        (
            (manifest.val_enrolled, manifest.val_unknown),
            reid_dm.validation_loaders_by_role(),
        ),
        (
            (manifest.test_enrolled, manifest.test_unknown),
            reid_dm.test_loaders_by_role(),
        ),
    ):
        enrolled, unknown = split_roles
        assert set(loaders) == {"gallery", "known_probes", "unknown_probes"}
        for f in loaders["gallery"].dataset.files:
            assert f.parent.parent.name == "train_amp"
            assert f.parent.name in enrolled
        for f in loaders["known_probes"].dataset.files:
            assert f.parent.parent.name == "test_amp"
            assert f.parent.name in enrolled
        for f in loaders["unknown_probes"].dataset.files:
            assert f.parent.parent.name == "test_amp"
            assert f.parent.name in unknown


def test_reid_train_files_only_from_training_identities(reid_dm):
    manifest = reid_dm.split_manifest
    for f in reid_dm.train_set.files:
        assert f.parent.parent.name == "train_amp"
        assert f.parent.name in manifest.train


def test_reid_eval_labels_are_stable_subject_ids(reid_dm):
    loaders = reid_dm.test_loaders_by_role()
    names = reid_dm.identity_names
    for role, loader in loaders.items():
        dataset = loader.dataset
        for f, label in zip(dataset.files, dataset.labels):
            assert names[label] == f.parent.name


def test_reid_val_dataloader_is_gallery_then_known_probes(reid_dm):
    val_loaders = reid_dm.val_dataloader()
    by_role = reid_dm.validation_loaders_by_role()
    assert len(val_loaders) == 2
    assert val_loaders[0].dataset.files == by_role["gallery"].dataset.files
    assert (
        val_loaders[1].dataset.files == by_role["known_probes"].dataset.files
    )


def test_reid_setup_is_idempotent(reid_dm):
    files_before = list(reid_dm.train_set.files)
    reid_dm.setup("fit")
    reid_dm.setup("test")
    assert list(reid_dm.train_set.files) == files_before


def test_reid_closed_set_loader_unchanged(fake_ntu_root):
    dm = data.build("ntu_fi_humanid", root=fake_ntu_root, batch_size=2)
    assert dm.num_classes == 14
    assert dm.task_type == "classification"
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (2, 3, 114, 500)
    assert y.dtype == torch.int64


@pytest.mark.data
@requires_ntu_humanid
def test_reid_real_data_contract():
    dm = data.build("ntu_fi_humanid_reid", root=DATA_ROOT, split_seed=42)
    assert len(dm.identity_names) == 14
    dm.setup("fit")
    dm.setup("test")
    total_train = sum(
        1
        for _ in (dm.root / "train_amp").glob("*/*.mat")
    )
    total_test = sum(1 for _ in (dm.root / "test_amp").glob("*/*.mat"))
    assert total_train == 294
    assert total_test == 546
    for loaders in (
        dm.validation_loaders_by_role(),
        dm.test_loaders_by_role(),
    ):
        for loader in loaders.values():
            x, y = next(iter(loader))
            assert x.shape[1:] == (3, 114, 500)
            assert y.dtype == torch.int64
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape[1:] == (3, 114, 500)
