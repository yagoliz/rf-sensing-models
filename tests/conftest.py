import os
from pathlib import Path

import pytest

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
