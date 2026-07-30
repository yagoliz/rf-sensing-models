"""Identity-disjoint Re-ID protocols and identity-balanced batch sampling.

An :class:`IdentitySplit` assigns every subject to exactly one role: encoder
training, validation gallery enrollment, validation unknown, test gallery
enrollment, or test unknown. Splits are generated from an explicit seed so
repeated experiments are reproducible and serializable.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

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
