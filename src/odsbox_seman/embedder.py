"""ONNX-based embedding engine for odsbox-seman.

Uses ``onnxruntime`` with the multilingual MiniLM model downloaded from the
Hugging Face Hub.  Embeddings are L2-normalised so cosine similarity reduces
to a dot product.

Device selection priority (fastest first):
  1. CUDA (NVIDIA GPU) via ``onnxruntime-gpu``
  2. DML (DirectML — Windows Copilot+ NPU/GPU) via ``onnxruntime-directml``
  3. CPU (always available, no extra install)
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import onnxruntime as ort

from odsbox_seman.types import SemanticSearchUnavailableError

log = logging.getLogger(__name__)

# Pre-quantized ONNX model exported by Xenova; the `model_quantized.onnx` file
# is ~50 MB and supports the full multilingual vocabulary.
_HF_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_ONNX_SUBFOLDER = "onnx"
_ONNX_FILENAME = "model.onnx"
_MAX_LENGTH = 512
_BATCH_SIZE = 64  # encode corpus in chunks to avoid OOM on large models


def _env_flag(name: str) -> bool:
    """Return True when an environment flag is enabled via common truthy values."""
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _detect_provider() -> list[str]:
    """Return the best available ONNX Runtime execution provider list."""
    force_cpu = _env_flag("ODSBOX_SEMAN_FORCE_CPU")
    allow_dml = _env_flag("ODSBOX_SEMAN_ALLOW_DML")

    if force_cpu:
        log.info("Semantic search: forcing CPUExecutionProvider via ODSBOX_SEMAN_FORCE_CPU=1.")
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort  # noqa: PLC0415

        available = ort.get_available_providers()
        windows = platform.system() == "Windows"

        if "CUDAExecutionProvider" in available:
            log.info("Semantic search: using CUDAExecutionProvider.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if (not windows or allow_dml) and "DmlExecutionProvider" in available:
            log.info("Semantic search: using DmlExecutionProvider (DirectML).")
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    except Exception:  # noqa: BLE001
        pass

    log.info("Semantic search: using CPUExecutionProvider.")
    return ["CPUExecutionProvider"]


def _normalize(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalise a batch of embedding vectors (in-place)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized: npt.NDArray[np.float32] = (embeddings / norms).astype(np.float32)
    return normalized


class _Tokenizer:
    """Wraps a ``tokenizers.Tokenizer`` and returns numpy arrays."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer  # noqa: PLC0415

        self._tok = Tokenizer.from_file(str(path))
        pad_id = self._tok.token_to_id("<pad>") or 0
        self._tok.enable_padding(pad_id=pad_id, pad_token="<pad>")
        self._tok.enable_truncation(max_length=_MAX_LENGTH)

    def __call__(self, texts: list[str]) -> dict[str, npt.NDArray[np.int64]]:
        encodings = self._tok.encode_batch(texts)
        return {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }


class OnnxEmbedder:
    """Loads the multilingual MiniLM ONNX model and encodes texts.

    The model and tokenizer are downloaded on first use and cached by the
    Hugging Face Hub library (default: ``~/.cache/huggingface/``).
    """

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._tokenizer: object | None = None
        self._model_path: Path | None = None
        self._providers: list[str] | None = None
        self._load_lock = Lock()

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Download (once) and initialise the ONNX model + tokenizer."""
        with self._load_lock:
            if self._session is not None:
                return

            try:
                from huggingface_hub import hf_hub_download  # noqa: PLC0415
            except ImportError as exc:
                raise _unavailable("huggingface-hub", "uv add huggingface-hub") from exc

            try:
                import onnxruntime as ort  # noqa: PLC0415
            except ImportError as exc:
                raise _unavailable("onnxruntime", "uv add onnxruntime") from exc

            try:
                import tokenizers  # noqa: F401, PLC0415
            except ImportError as exc:
                raise _unavailable("tokenizers", "uv add tokenizers") from exc

            def fetch(filename: str) -> Path:
                # Cache first: a network call (and its HF Hub warnings) is only needed on first use.
                try:
                    return Path(
                        hf_hub_download(repo_id=_HF_REPO, filename=filename, local_files_only=True)
                    )
                except Exception:  # noqa: BLE001
                    log.info("Downloading %s from HF Hub: %s …", filename, _HF_REPO)
                    return Path(hf_hub_download(repo_id=_HF_REPO, filename=filename))

            model_path = fetch(f"{_ONNX_SUBFOLDER}/{_ONNX_FILENAME}")

            providers = _detect_provider()
            self._model_path = model_path
            self._providers = providers
            try:
                session = ort.InferenceSession(str(model_path), providers=providers)
            except Exception:  # noqa: BLE001
                log.warning(
                    "Failed to initialise ONNX model with providers %s; retrying on CPU.",
                    providers,
                )
                session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                self._providers = ["CPUExecutionProvider"]
            tokenizer = _Tokenizer(fetch("tokenizer.json"))
            self._tokenizer = tokenizer
            self._session = session
            log.info("ONNX model loaded from %s with providers %s", model_path, self._providers)

    def _fallback_to_cpu(self) -> None:
        """Retry the model on CPU when a GPU provider becomes unstable at runtime."""
        if self._model_path is None or self._session is None:
            return

        if self._providers == ["CPUExecutionProvider"]:
            return

        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            return

        log.warning(
            "ONNX inference failed with providers %s; retrying on CPU.",
            self._providers,
        )
        self._session = ort.InferenceSession(
            str(self._model_path), providers=["CPUExecutionProvider"]
        )
        self._providers = ["CPUExecutionProvider"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], *, normalize: bool = True) -> npt.NDArray[np.float32]:
        """Encode *texts* into embedding vectors.

        Parameters
        ----------
        texts:
            List of strings to encode.
        normalize:
            If ``True`` (default), return L2-normalised embeddings.

        Returns
        -------
        numpy.ndarray of shape ``(len(texts), embedding_dim)`` and dtype
        ``float32``.
        """
        self._ensure_loaded()
        assert self._tokenizer is not None  # noqa: S101
        assert self._session is not None  # noqa: S101

        # Encode in batches to avoid OOM with large corpora.
        if len(texts) > _BATCH_SIZE:
            chunks = [
                self._encode_batch(texts[i : i + _BATCH_SIZE], normalize=False)
                for i in range(0, len(texts), _BATCH_SIZE)
            ]
            pooled = np.concatenate(chunks, axis=0).astype(np.float32)
            return _normalize(pooled) if normalize else pooled

        return self._encode_batch(texts, normalize=normalize)

    def _encode_batch(self, texts: list[str], *, normalize: bool = True) -> npt.NDArray[np.float32]:
        """Encode a single batch of texts (no further chunking)."""
        assert self._tokenizer is not None  # noqa: S101
        assert self._session is not None  # noqa: S101

        # Tokenise
        encoded = self._tokenizer(texts)  # type: ignore[operator]

        # Run ONNX inference

        session: ort.InferenceSession = self._session
        input_names = {inp.name for inp in session.get_inputs()}

        input_ids: npt.NDArray[np.int64] = encoded["input_ids"].astype(np.int64)
        ort_inputs: dict[str, npt.NDArray[np.int64]] = {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in input_names:
            # Supply token_type_ids from the tokenizer when available; otherwise
            # fall back to all-zeros (correct for single-sentence BERT inputs).
            if "token_type_ids" in encoded:
                ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
            else:
                ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

        try:
            outputs = session.run(None, ort_inputs)
        except Exception:  # noqa: BLE001
            self._fallback_to_cpu()
            assert self._session is not None  # noqa: S101
            outputs = self._session.run(None, ort_inputs)

        # Mean-pool the last hidden state over non-padding tokens
        last_hidden: npt.NDArray[np.float32] = outputs[0]
        attention_mask: npt.NDArray[np.float32] = encoded["attention_mask"].astype(np.float32)
        mask_expanded = attention_mask[:, :, np.newaxis]
        sum_hidden = (last_hidden * mask_expanded).sum(axis=1)
        sum_mask = mask_expanded.sum(axis=1).clip(min=1e-9)
        pooled: npt.NDArray[np.float32] = (sum_hidden / sum_mask).astype(np.float32)

        if normalize:
            pooled = _normalize(pooled)
        return pooled


def _unavailable(package: str, install_cmd: str) -> SemanticSearchUnavailableError:
    return SemanticSearchUnavailableError(
        f"Semantic search requires '{package}'. Install it with: {install_cmd}"
    )
