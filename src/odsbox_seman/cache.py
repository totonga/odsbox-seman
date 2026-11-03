"""Disk cache for ODS model search embeddings.

Cache files are stored in ``~/.ods-seman/search_cache/`` (or a custom directory
passed to :class:`CacheManager`) and keyed by a SHA-256 hash of the serialised
ODS model protobuf.  Each cache entry consists of two files:

* ``<hash>.npz``   — the embedding matrix (float32, shape ``(n, dim)``) and the
  per-entry ``alias_flags`` (float32, shape ``(n,)``; 1 for alias entries)
* ``<hash>.json``  — corpus metadata list (text + ModelMatch fields)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    from odsbox.proto.ods_pb2 import Model

    from odsbox_seman.config import ModelConfig
    from odsbox_seman.types import ModelMatch

log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR: Path = Path.home() / ".ods-seman" / "search_cache"
_CACHE_DIR_ENV_VAR = "ODSBOX_SEMAN_CACHE_DIR"
_ENTRY_LOCKS: dict[tuple[str, str], RLock] = {}
_ENTRY_LOCKS_GUARD = Lock()

# Bump when the corpus text construction changes so old caches are not reused.
_CORPUS_FORMAT_VERSION = 3


def model_hash(model: Model, config: ModelConfig | None = None) -> str:
    """Return a hex SHA-256 digest of the serialised model plus the config fingerprint."""
    sha = hashlib.sha256(model.SerializeToString())
    sha.update(str(_CORPUS_FORMAT_VERSION).encode("ascii"))
    if config is not None:
        sha.update(config.fingerprint().encode("ascii"))
    return sha.hexdigest()


class CacheManager:
    """Manages loading and saving of embedding caches.

    Cache transactions are synchronized per directory and digest across instances
    within one process. Separate processes are not coordinated.

    Parameters
    ----------
    cache_dir:
        Directory where cache files are stored. Created on first write. If omitted,
        ``ODSBOX_SEMAN_CACHE_DIR`` is used when set; otherwise the default is
        ``~/.ods-seman/search_cache/``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        env_cache_dir = os.environ.get(_CACHE_DIR_ENV_VAR)
        if cache_dir is not None:
            self._dir = cache_dir
        elif env_cache_dir:
            self._dir = Path(env_cache_dir).expanduser()
        else:
            self._dir = _DEFAULT_CACHE_DIR

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

    @contextmanager
    def lock(self, digest: str) -> Generator[None]:
        """Serialize a cache transaction for this directory/key within this process."""
        key = (os.path.normcase(str(self._dir.resolve())), digest)
        with _ENTRY_LOCKS_GUARD:
            entry_lock = _ENTRY_LOCKS.get(key)
            if entry_lock is None:
                entry_lock = RLock()
                _ENTRY_LOCKS[key] = entry_lock
        with entry_lock:
            yield

    def save(
        self,
        digest: str,
        embeddings: npt.NDArray[np.float32],
        corpus: list[tuple[str, ModelMatch]],
        alias_flags: npt.NDArray[np.float32] | None = None,
    ) -> None:
        """Persist *embeddings*, *corpus* metadata and *alias_flags* to disk."""
        with self.lock(digest):
            self._dir.mkdir(parents=True, exist_ok=True)
            npz_path = self._npz_path(digest)
            json_path = self._json_path(digest)

            if alias_flags is None:
                alias_flags = np.zeros(len(corpus), dtype=np.float32)
            np.savez_compressed(str(npz_path), embeddings=embeddings, alias_flags=alias_flags)

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
    ) -> (
        tuple[npt.NDArray[np.float32], list[tuple[str, ModelMatch]], npt.NDArray[np.float32]] | None
    ):
        """Load cached embeddings and corpus, or return ``None`` on any problem.

        Parameters
        ----------
        digest:
            SHA-256 hash that identifies the cache entry.
        expected_length:
            Number of corpus entries we expect; mismatches cause a cache miss.
        """
        from odsbox_seman.types import ModelMatch  # noqa: PLC0415

        with self.lock(digest):
            npz_path = self._npz_path(digest)
            json_path = self._json_path(digest)

            if not npz_path.exists() or not json_path.exists():
                return None

            try:
                with np.load(str(npz_path)) as data:
                    embeddings: npt.NDArray[np.float32] = data["embeddings"]
                    alias_flags: npt.NDArray[np.float32] = data["alias_flags"]

                raw: list[dict[str, object]] = json.loads(json_path.read_text(encoding="utf-8"))

                if len(raw) != len(embeddings) or len(alias_flags) != len(embeddings):
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
                return embeddings, corpus, alias_flags
            except Exception:  # noqa: BLE001
                log.warning("Failed to load search cache — will rebuild.", exc_info=True)
                return None
