"""Shared pytest fixtures for odsbox-seman tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

_DATA_DIR = Path(__file__).parent / "data"
_NVH_MODEL_PATH = _DATA_DIR / "mdm_nvh_model.json"


@pytest.fixture(scope="session")
def nvh_model():  # type: ignore[no-untyped-def]
    """Load the NVH model protobuf once per test session."""
    from google.protobuf.json_format import Parse
    from odsbox.proto.ods_pb2 import Model

    model: Model = Model()
    assert _NVH_MODEL_PATH.exists(), f"Model file not found: {_NVH_MODEL_PATH}"
    with _NVH_MODEL_PATH.open(encoding="utf-8") as fh:
        Parse(fh.read(), model)
    return model


@pytest.fixture()
def fake_embeddings():  # type: ignore[no-untyped-def]
    """Factory for synthetic float32 embedding arrays."""

    def _make(n: int, dim: int = 32) -> npt.NDArray[np.float32]:
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalized: npt.NDArray[np.float32] = (raw / norms).astype(np.float32)
        return normalized

    return _make
