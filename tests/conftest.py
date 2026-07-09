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