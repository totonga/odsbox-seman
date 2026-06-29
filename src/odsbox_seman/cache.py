"""Disk cache for ODS model search embeddings.

Cache files are stored in ``~/.ods-seman/search_cache/`` (or a custom directory
passed to :class:`CacheManager`) and keyed by a SHA-256 hash of the serialised
ODS model protobuf.  Each cache entry consists of two files:

* ``<hash>.npz``   — the embedding matrix (float32, shape ``(n, dim)``)
* ``<hash>.json``  — corpus metadata list (text + ModelMatch fields)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    from odsbox.proto.ods_pb2 import Model

    from odsbox_seman.types import ModelMatch

log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR: Path = Path.home() / ".ods-seman" / "search_cache"


def model_hash(model: Model) -> str:
    """Return a hex SHA-256 digest of the serialised model protobuf."""
    return hashlib.sha256(model.SerializeToString()).hexdigest()


class CacheManager:
    """Manages loading and saving of embedding caches.

    Parameters
    ----------
    cache_dir:
        Directory where cache files are stored.  Created on first write.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir: Path = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _npz_path(self, digest: str) -> Path:
        return self._dir / f"{digest}.npz"

    def _json_path(self, digest: str) -> Path:
        return self._dir / f"{digest}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        digest: str,
        embeddings: npt.NDArray[np.float32],
        corpus: list[tuple[str, ModelMatch]],
    ) -> None:
        """Persist *embeddings* and *corpus* metadata to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)
        npz_path = self._npz_path(digest)
        json_path = self._json_path(digest)

        np.savez_compressed(str(npz_path), embeddings=embeddings)

        raw: list[dict[str, object]] = []
        for text, match in corpus:
            entry = match.to_dict()
            entry["text"] = text
            raw.append(entry)

        json_path.write_text(json.dumps(raw), encoding="utf-8")
        log.info("Saved search cache → %s", npz_path)

    def load(
        self,
        digest: str,
        expected_length: int,
    ) -> tuple[npt.NDArray[np.float32], list[tuple[str, ModelMatch]]] | None:
        """Load cached embeddings and corpus, or return ``None`` on any problem.

        Parameters
        ----------
        digest:
            SHA-256 hash that identifies the cache entry.
        expected_length:
            Number of corpus entries we expect; mismatches cause a cache miss.
        """
        from odsbox_seman.types import ModelMatch  # noqa: PLC0415

        npz_path = self._npz_path(digest)
        json_path = self._json_path(digest)

        if not npz_path.exists() or not json_path.exists():
            return None

        try:
            data = np.load(str(npz_path))
            embeddings: npt.NDArray[np.float32] = data["embeddings"]

            raw: list[dict[str, object]] = json.loads(json_path.read_text(encoding="utf-8"))

            if len(raw) != len(embeddings):
                log.warning(
                    "Cache length mismatch (%d vs %d) — rebuilding.",
                    len(raw),
                    len(embeddings),
                )
                return None

            if len(raw) != expected_length:
                log.warning(
                    "Cache expected length mismatch (%d vs %d) — rebuilding.",
                    expected_length,
                    len(raw),
                )
                return None

            corpus: list[tuple[str, ModelMatch]] = []
            for item in raw:
                corpus.append(
                    (
                        str(item["text"]),
                        ModelMatch(
                            kind=item["kind"],  # type: ignore[arg-type]
                            entity_name=str(item["entity_name"]),
                            entity_base_name=str(item["entity_base_name"]),
                            item_name=str(item["item_name"]),
                            item_base_name=str(item["item_base_name"]),
                            data_type=int(str(item["data_type"])),
                            score=0.0,
                        ),
                    )
                )
            log.info("Loaded search cache from %s (%d entries).", npz_path, len(corpus))
            return embeddings, corpus

        except Exception:  # noqa: BLE001
            log.warning("Failed to load search cache — will rebuild.", exc_info=True)
            return None
