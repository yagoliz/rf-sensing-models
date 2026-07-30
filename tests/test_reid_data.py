import pytest

from rfsensing.data.reid import IdentitySplit, make_identity_split

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
