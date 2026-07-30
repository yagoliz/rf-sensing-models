"""Identity-disjoint Re-ID protocols and identity-balanced batch sampling.

An :class:`IdentitySplit` assigns every subject to exactly one role: encoder
training, validation gallery enrollment, validation unknown, test gallery
enrollment, or test unknown. Splits are generated from an explicit seed so
repeated experiments are reproducible and serializable.
"""

import math
from dataclasses import asdict, dataclass
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

_ROLE_FIELDS = (
    "train",
    "val_enrolled",
    "val_unknown",
    "test_enrolled",
    "test_unknown",
)


@dataclass(frozen=True)
class IdentitySplit:
    seed: int
    train: tuple[str, ...]
    val_enrolled: tuple[str, ...]
    val_unknown: tuple[str, ...]
    test_enrolled: tuple[str, ...]
    test_unknown: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        for role in _ROLE_FIELDS:
            d[role] = list(d[role])
        return d

    def validate(self) -> None:
        for role in _ROLE_FIELDS:
            ids = getattr(self, role)
            if not ids:
                raise ValueError(f"role {role!r} must contain at least 1 identity")
            if len(set(ids)) != len(ids):
                raise ValueError(f"role {role!r} contains duplicate identities")
        for i, a in enumerate(_ROLE_FIELDS):
            for b in _ROLE_FIELDS[i + 1 :]:
                overlap = set(getattr(self, a)) & set(getattr(self, b))
                if overlap:
                    raise ValueError(
                        f"roles {a!r} and {b!r} overlap: {sorted(overlap)}"
                    )


def make_identity_split(
    identities: Sequence[str],
    *,
    seed: int,
    train_count: int = 7,
    val_enrolled_count: int = 2,
    val_unknown_count: int = 1,
    test_enrolled_count: int = 3,
    test_unknown_count: int = 1,
) -> IdentitySplit:
    """Deterministically partition ``identities`` into disjoint Re-ID roles."""
    if len(set(identities)) != len(identities):
        raise ValueError("identities contain duplicates")
    counts = {
        "train": train_count,
        "val_enrolled": val_enrolled_count,
        "val_unknown": val_unknown_count,
        "test_enrolled": test_enrolled_count,
        "test_unknown": test_unknown_count,
    }
    for role, count in counts.items():
        if count < 1:
            raise ValueError(f"role {role!r} must contain at least 1 identity")
    total = sum(counts.values())
    if total != len(identities):
        raise ValueError(
            f"role counts sum to {total} but {len(identities)} identities "
            "are available"
        )
    # Sort before shuffling so filesystem iteration order cannot alter a split.
    pool = sorted(identities)
    np.random.default_rng(seed).shuffle(pool)
    roles: dict[str, tuple[str, ...]] = {}
    start = 0
    for role, count in counts.items():
        roles[role] = tuple(pool[start : start + count])
        start += count
    split = IdentitySplit(seed=seed, **roles)
    split.validate()
    return split


class IdentityBatchSampler(Sampler[list[int]]):
    """P×K batch sampler: P distinct identities, K samples per identity.

    Batches are deterministic given ``seed`` and the epoch set through
    :meth:`set_epoch`. Identities with fewer than K samples are sampled with
    replacement; otherwise indices within a batch are drawn without
    replacement.
    """

    def __init__(
        self,
        labels: Sequence[int],
        identities_per_batch: int,
        samples_per_identity: int,
        *,
        seed: int = 42,
    ):
        if len(labels) == 0:
            raise ValueError("labels must not be empty")
        if any(int(label) != label for label in labels):
            raise ValueError("labels must be integer identity indices")
        if identities_per_batch < 2:
            raise ValueError(
                f"identities_per_batch must be >= 2, got {identities_per_batch}"
            )
        if samples_per_identity < 2:
            raise ValueError(
                f"samples_per_identity must be >= 2, got {samples_per_identity}"
            )
        self._indices_by_label: dict[int, np.ndarray] = {}
        for index, label in enumerate(labels):
            self._indices_by_label.setdefault(int(label), []).append(index)
        if len(self._indices_by_label) < identities_per_batch:
            raise ValueError(
                f"need at least {identities_per_batch} distinct identities, "
                f"got {len(self._indices_by_label)}"
            )
        self._indices_by_label = {
            label: np.asarray(indices)
            for label, indices in self._indices_by_label.items()
        }
        self.identities_per_batch = identities_per_batch
        self.samples_per_identity = samples_per_identity
        self.seed = seed
        self._epoch = 0
        batch = identities_per_batch * samples_per_identity
        self._num_batches = max(1, math.ceil(len(labels) / batch))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        identities = sorted(self._indices_by_label)
        for _ in range(self._num_batches):
            chosen = rng.choice(
                len(identities), size=self.identities_per_batch, replace=False
            )
            batch: list[int] = []
            for identity_pos in chosen:
                pool = self._indices_by_label[identities[identity_pos]]
                replace = len(pool) < self.samples_per_identity
                picks = rng.choice(
                    pool, size=self.samples_per_identity, replace=replace
                )
                batch.extend(int(i) for i in picks)
            yield batch
