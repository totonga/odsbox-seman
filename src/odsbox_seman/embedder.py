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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from odsbox_seman.types import SemanticSearchUnavailableError

if TYPE_CHECKING:
    import numpy.typing as npt

log = logging.getLogger(__name__)

# Pre-quantized ONNX model exported by Xenova; the `model_quantized.onnx` file
# is ~50 MB and supports the full multilingual vocabulary.
_HF_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_ONNX_SUBFOLDER = "onnx"
_ONNX_FILENAME = "model.onnx"
_MAX_LENGTH = 512
_BATCH_SIZE = 64  # encode corpus in chunks to avoid OOM on large models


def _detect_provider() -> list[str]:
    """Return the best available ONNX Runtime execution provider list."""
    try:
        import onnxruntime as ort  # noqa: PLC0415

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            log.info("Semantic search: using CUDAExecutionProvider.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if "DmlExecutionProvider" in available:
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
    return (embeddings / norms).astype(np.float32)


class OnnxEmbedder:
    """Loads the multilingual MiniLM ONNX model and encodes texts.

    The model and tokenizer are downloaded on first use and cached by the
    Hugging Face Hub library (default: ``~/.cache/huggingface/``).
    """

    def __init__(self) -> None:
        self._session: object | None = None
        self._tokenizer: object | None = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Download (once) and initialise the ONNX model + tokenizer."""
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
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise _unavailable("transformers", "uv add transformers") from exc

        log.info("Downloading/loading ONNX model from HF Hub: %s …", _HF_REPO)
        model_path: Path = Path(
            hf_hub_download(
                repo_id=_HF_REPO,
                filename=f"{_ONNX_SUBFOLDER}/{_ONNX_FILENAME}",
            )
        )

        providers = _detect_provider()
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._tokenizer = AutoTokenizer.from_pretrained(_HF_REPO)
        log.info("ONNX model loaded from %s", model_path)

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
        encoded = self._tokenizer(  # type: ignore[operator]
            texts,
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="np",
        )

        # Run ONNX inference
        import onnxruntime as ort  # noqa: PLC0415

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

        outputs = session.run(None, ort_inputs)

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
