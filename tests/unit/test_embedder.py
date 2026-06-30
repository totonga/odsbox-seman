"""Unit tests for OnnxEmbedder and device detection."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from odsbox_seman.embedder import _detect_provider, _normalize
from odsbox_seman.types import SemanticSearchUnavailableError


class TestNormalize:
    def test_unit_vectors_unchanged(self) -> None:
        v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = _normalize(v)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0, 1.0], atol=1e-6)

    def test_normalizes_arbitrary_vectors(self) -> None:
        v = np.array([[3.0, 4.0]], dtype=np.float32)  # norm = 5
        result = _normalize(v)
        np.testing.assert_allclose(result, [[0.6, 0.8]], atol=1e-6)

    def test_batch_normalized(self) -> None:
        rng = np.random.default_rng(1)
        v = rng.standard_normal((20, 64)).astype(np.float32)
        result = _normalize(v)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(20), atol=1e-5)

    def test_zero_vector_handled(self) -> None:
        v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        result = _normalize(v)
        # Should not produce NaN or inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_returns_float32(self) -> None:
        v = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result = _normalize(v.astype(np.float32))
        assert result.dtype == np.float32


class TestDetectProvider:
    def test_returns_cpu_by_default(self, mocker: Any) -> None:
        """When no GPU providers are available, CPU is returned."""
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        mocker.patch.dict("sys.modules", {"onnxruntime": mock_ort})
        providers = _detect_provider()
        assert "CPUExecutionProvider" in providers

    def test_prefers_cuda_when_available(self, mocker: Any) -> None:
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mocker.patch.dict("sys.modules", {"onnxruntime": mock_ort})
        providers = _detect_provider()
        assert providers[0] == "CUDAExecutionProvider"
        assert "CPUExecutionProvider" in providers

    def test_prefers_dml_over_cpu(self, mocker: Any) -> None:
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        mocker.patch.dict("sys.modules", {"onnxruntime": mock_ort})
        providers = _detect_provider()
        assert providers[0] == "DmlExecutionProvider"
        assert "CPUExecutionProvider" in providers

    def test_falls_back_to_cpu_on_import_error(self, mocker: Any) -> None:
        """If onnxruntime import fails during detection, fall back to CPU list."""
        mocker.patch.dict("sys.modules", {"onnxruntime": None})
        providers = _detect_provider()
        assert providers == ["CPUExecutionProvider"]

    def test_cuda_takes_priority_over_dml(self, mocker: Any) -> None:
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        mocker.patch.dict("sys.modules", {"onnxruntime": mock_ort})
        providers = _detect_provider()
        assert providers[0] == "CUDAExecutionProvider"


class TestOnnxEmbedderInit:
    def test_ensure_loaded_raises_when_onnxruntime_missing(self, mocker: Any) -> None:
        from odsbox_seman.embedder import OnnxEmbedder

        mocker.patch.dict(
            "sys.modules",
            {
                "onnxruntime": None,
                "huggingface_hub": MagicMock(),
                "transformers": MagicMock(),
            },
        )
        embedder = OnnxEmbedder()
        with pytest.raises(SemanticSearchUnavailableError, match="onnxruntime"):
            embedder._ensure_loaded()

    def test_ensure_loaded_raises_when_transformers_missing(self, mocker: Any) -> None:
        from odsbox_seman.embedder import OnnxEmbedder

        mock_hub = MagicMock()
        mock_hub.hf_hub_download.return_value = "/fake/path/model.onnx"

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = MagicMock()

        mocker.patch.dict(
            "sys.modules",
            {
                "huggingface_hub": mock_hub,
                "onnxruntime": mock_ort,
                "transformers": None,
            },
        )
        embedder = OnnxEmbedder()
        with pytest.raises(SemanticSearchUnavailableError, match="transformers"):
            embedder._ensure_loaded()

    def test_ensure_loaded_raises_when_hf_hub_missing(self, mocker: Any) -> None:
        from odsbox_seman.embedder import OnnxEmbedder

        mocker.patch.dict(
            "sys.modules",
            {
                "huggingface_hub": None,
                "onnxruntime": MagicMock(),
                "transformers": MagicMock(),
            },
        )
        embedder = OnnxEmbedder()
        with pytest.raises(SemanticSearchUnavailableError, match="huggingface-hub"):
            embedder._ensure_loaded()

    def test_encode_calls_session(self, mocker: Any) -> None:
        """Verify encode() runs tokenizer + ONNX session and returns float32 array."""
        from odsbox_seman.embedder import OnnxEmbedder

        # Fake tokenizer output
        fake_input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        fake_attention = np.array([[1, 1, 1]], dtype=np.int64)
        mock_tokenizer = MagicMock(
            return_value={
                "input_ids": fake_input_ids,
                "attention_mask": fake_attention,
            }
        )

        # Fake ONNX session output: shape (1, seq, dim)
        fake_hidden = np.ones((1, 3, 16), dtype=np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [fake_hidden]
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]

        embedder = OnnxEmbedder()
        embedder._session = mock_session
        embedder._tokenizer = mock_tokenizer

        result = embedder.encode(["hello world"])

        assert result.shape == (1, 16)
        assert result.dtype == np.float32
        mock_session.run.assert_called_once()

    def test_encode_without_normalize(self, mocker: Any) -> None:
        from odsbox_seman.embedder import OnnxEmbedder

        fake_input_ids = np.array([[1, 2]], dtype=np.int64)
        fake_attention = np.array([[1, 1]], dtype=np.int64)
        mock_tokenizer = MagicMock(
            return_value={
                "input_ids": fake_input_ids,
                "attention_mask": fake_attention,
            }
        )
        fake_hidden = np.array([[[2.0, 4.0, 0.0]]], dtype=np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [fake_hidden]
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]

        embedder = OnnxEmbedder()
        embedder._session = mock_session
        embedder._tokenizer = mock_tokenizer

        result = embedder.encode(["hi"], normalize=False)
        # Mean pool of [[2,4,0]] over 2 tokens (both masked) = [[2,4,0]]
        assert result.shape == (1, 3)
        # Not normalized — norm should not be 1
        norm = np.linalg.norm(result[0])
        assert not np.isclose(norm, 1.0)

    def test_encode_with_token_type_ids(self, mocker: Any) -> None:
        """If model has token_type_ids input, they must be passed."""
        from odsbox_seman.embedder import OnnxEmbedder

        fake_input_ids = np.array([[1, 2]], dtype=np.int64)
        fake_attention = np.array([[1, 1]], dtype=np.int64)
        fake_token_types = np.array([[0, 0]], dtype=np.int64)
        mock_tokenizer = MagicMock(
            return_value={
                "input_ids": fake_input_ids,
                "attention_mask": fake_attention,
                "token_type_ids": fake_token_types,
            }
        )
        fake_hidden = np.ones((1, 2, 4), dtype=np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [fake_hidden]

        # MagicMock(name=...) sets the mock's display name, not the .name attribute.
        # We must assign .name explicitly so the embedder's input-name set works correctly.
        inp1, inp2, inp3 = MagicMock(), MagicMock(), MagicMock()
        inp1.name = "input_ids"
        inp2.name = "attention_mask"
        inp3.name = "token_type_ids"
        mock_session.get_inputs.return_value = [inp1, inp2, inp3]

        embedder = OnnxEmbedder()
        embedder._session = mock_session
        embedder._tokenizer = mock_tokenizer

        result = embedder.encode(["test"])
        call_kwargs = mock_session.run.call_args[0][1]
        assert "token_type_ids" in call_kwargs
        assert result.shape == (1, 4)

    def test_encode_token_type_ids_zeros_fallback(self, mocker: Any) -> None:
        """When model requires token_type_ids but tokenizer doesn't return them, zeros are used."""
        from odsbox_seman.embedder import OnnxEmbedder

        fake_input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        fake_attention = np.array([[1, 1, 1]], dtype=np.int64)
        # Tokenizer does NOT return token_type_ids
        mock_tokenizer = MagicMock(
            return_value={
                "input_ids": fake_input_ids,
                "attention_mask": fake_attention,
            }
        )
        fake_hidden = np.ones((1, 3, 8), dtype=np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [fake_hidden]

        inp1, inp2, inp3 = MagicMock(), MagicMock(), MagicMock()
        inp1.name = "input_ids"
        inp2.name = "attention_mask"
        inp3.name = "token_type_ids"
        mock_session.get_inputs.return_value = [inp1, inp2, inp3]

        embedder = OnnxEmbedder()
        embedder._session = mock_session
        embedder._tokenizer = mock_tokenizer

        result = embedder.encode(["test"])
        call_kwargs = mock_session.run.call_args[0][1]
        # Must have been filled with zeros fallback
        assert "token_type_ids" in call_kwargs
        np.testing.assert_array_equal(
            call_kwargs["token_type_ids"], np.zeros((1, 3), dtype=np.int64)
        )
        assert result.shape == (1, 8)

    def test_encode_batching_splits_large_input(self, mocker: Any) -> None:
        """When len(texts) > _BATCH_SIZE, encode() splits into chunks and concatenates."""
        from odsbox_seman.embedder import _BATCH_SIZE, OnnxEmbedder

        n = _BATCH_SIZE + 5  # just over the batch boundary
        texts = [f"text {i}" for i in range(n)]
        dim = 8

        # Each call to the tokenizer/session returns a unit-vector batch
        def make_tokenizer_output(batch: list[str]) -> dict[str, np.ndarray]:
            b = len(batch)
            return {
                "input_ids": np.ones((b, 4), dtype=np.int64),
                "attention_mask": np.ones((b, 4), dtype=np.int64),
            }

        call_count = 0

        def tokenizer_side_effect(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
            nonlocal call_count
            batch_texts = args[0] if args else kwargs.get("text", [])
            call_count += 1
            return make_tokenizer_output(batch_texts)  # type: ignore[arg-type]

        mock_tokenizer = MagicMock(side_effect=tokenizer_side_effect)

        def session_run_side_effect(
            output_names: object,
            inputs: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            b = inputs["input_ids"].shape[0]
            return [np.ones((b, 4, dim), dtype=np.float32)]

        mock_session = MagicMock()
        mock_session.run.side_effect = session_run_side_effect
        mock_session.get_inputs.return_value = []

        embedder = OnnxEmbedder()
        embedder._session = mock_session
        embedder._tokenizer = mock_tokenizer

        result = embedder.encode(texts, normalize=False)
        assert result.shape == (n, dim)
        # Should have been called at least twice (two chunks)
        assert call_count >= 2
