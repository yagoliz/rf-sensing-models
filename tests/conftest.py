import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

DATA_ROOT = Path(
    os.environ.get(
        "RFSENSING_DATA", Path(__file__).resolve().parents[3] / "data"
    )
)

requires_data = pytest.mark.skipif(
    not DATA_ROOT.exists(), reason=f"dataset root not found: {DATA_ROOT}"
)

WIMANS_ROOT = DATA_ROOT / "WiMANS"
WIMANS_AMP = (
    WIMANS_ROOT / "wifi_csi" / "amp"
    if (WIMANS_ROOT / "wifi_csi" / "amp").is_dir()
    else WIMANS_ROOT / "amp"
)
WIMANS_COMPLETE = (
    (WIMANS_ROOT / "annotation.csv").is_file()
    and WIMANS_AMP.is_dir()
    and len(list(WIMANS_AMP.glob("*.npy"))) == 11286
)
requires_wimans = pytest.mark.skipif(
    not WIMANS_COMPLETE,
    reason=f"complete WiMANS amplitude dataset not found under {WIMANS_ROOT}",
)

NTU_HUMANID_ROOT = DATA_ROOT / "NTU-Fi-HumanID"
NTU_HUMANID_COMPLETE = (
    len(list((NTU_HUMANID_ROOT / "train_amp").glob("*/*.mat"))) == 294
    and len(list((NTU_HUMANID_ROOT / "test_amp").glob("*/*.mat"))) == 546
)
requires_ntu_humanid = pytest.mark.skipif(
    not NTU_HUMANID_COMPLETE,
    reason=f"complete NTU-Fi HumanID dataset not found under {NTU_HUMANID_ROOT}",
)


def make_ntu_humanid_tree(
    root, num_identities=14, train_samples=3, test_samples=2
):
    """Write a small NTU-Fi-HumanID-shaped MAT tree under ``root``."""
    base = Path(root) / "NTU-Fi-HumanID"
    rng = np.random.default_rng(0)
    for split, samples in (
        ("train_amp", train_samples),
        ("test_amp", test_samples),
    ):
        for identity in range(num_identities):
            subject_dir = base / split / f"{identity + 1:03d}"
            subject_dir.mkdir(parents=True, exist_ok=True)
            for sample in range(samples):
                # Integer-valued amplitudes around the NTU-Fi mean keep the
                # compressed MAT files small while staying shape-valid.
                amplitude = rng.integers(
                    38, 46, size=(342, 2000)
                ).astype(np.float32) + 0.5 * identity
                sio.savemat(
                    subject_dir / f"sample_{sample}.mat",
                    {"CSIamp": amplitude},
                    do_compression=True,
                )
    return root


@pytest.fixture(scope="session")
def fake_ntu_root(tmp_path_factory):
    """Session-scoped generated NTU-Fi HumanID tree (14 identities)."""
    return make_ntu_humanid_tree(tmp_path_factory.mktemp("fake_ntu"))
