"""Semantic search index over an ASAM ODS model.

Uses ONNX Runtime with the multilingual MiniLM model to produce
L2-normalised embeddings; cosine similarity is computed as a dot product.

Embeddings are cached on disk (``~/.ods-seman/search_cache/``) keyed by a
SHA-256 hash of the serialised model protobuf so the index is only rebuilt
when the server model changes.

Example::

    from odsbox_seman import ModelSearcher

    searcher = ModelSearcher(my_ods_model)
    results = searcher.search("tyre pressure sensor", top_k=10)
    for r in results:
        print(r.entity_name, r.item_name, r.score)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from odsbox_seman.cache import CacheManager, model_hash
from odsbox_seman.embedder import OnnxEmbedder
from odsbox_seman.tokenizer import tokenize
from odsbox_seman.types import ModelMatch, SemanticSearchUnavailableError

if TYPE_CHECKING:
    import numpy.typing as npt
    from odsbox.proto.ods_pb2 import Model

log = logging.getLogger(__name__)

# DataTypeEnum value for DT_DATE in the ODS protobuf spec.
_DT_DATE = 10


class ModelSearcher:
    """Semantic search index built from an ASAM ODS :class:`~odsbox.proto.ods_pb2.Model`.

    Parameters
    ----------
    model:
        Parsed ODS model protobuf.
    cache_dir:
        Custom cache directory.  Defaults to ``~/.ods-seman/search_cache/``.
    """

    def __init__(
        self,
        model: Model,
        cache_dir: Path | None = None,
    ) -> None:
        self._model = model
        self._digest: str = model_hash(model)
        self._cache = CacheManager(cache_dir)
        self._corpus: list[tuple[str, ModelMatch]] = self._build_corpus()
        self._embeddings: npt.NDArray[np.float32] | None = None
        self._embedder: OnnxEmbedder | None = None

    # ------------------------------------------------------------------
    # Corpus construction
    # ------------------------------------------------------------------

    def _build_corpus(self) -> list[tuple[str, ModelMatch]]:
        """Build the list of ``(text, ModelMatch)`` pairs from the ODS model."""
        entries: list[tuple[str, ModelMatch]] = []

        for entity in self._model.entities.values():
            e_text = f"{tokenize(entity.name)} {tokenize(entity.base_name)}"

            for attr in entity.attributes.values():
                text = f"{e_text} {tokenize(attr.name)} {tokenize(attr.base_name)}"
                entries.append(
                    (
                        text,
                        ModelMatch(
                            kind="attribute",
                            entity_name=entity.name,
                            entity_base_name=entity.base_name,
                            item_name=attr.name,
                            item_base_name=attr.base_name,
                            data_type=int(attr.data_type),
                            score=0.0,
                        ),
                    )
                )

            for rel in entity.relations.values():
                text = (
                    f"{e_text} {tokenize(rel.name)} {tokenize(rel.base_name)} "
                    f"{tokenize(rel.entity_name)} {tokenize(rel.entity_base_name)} "
                    f"{tokenize(rel.inverse_name)} {tokenize(rel.inverse_base_name)}"
                )
                entries.append(
                    (
                        text,
                        ModelMatch(
                            kind="relation",
                            entity_name=entity.name,
                            entity_base_name=entity.base_name,
                            item_name=rel.name,
                            item_base_name=rel.base_name,
                            data_type=0,
                            score=0.0,
                        ),
                    )
                )

        for enum in self._model.enumerations.values():
            items_text = " ".join(tokenize(k) for k in enum.items)
            text = f"{tokenize(enum.name)} {items_text}"
            entries.append(
                (
                    text,
                    ModelMatch(
                        kind="enumeration",
                        entity_name="",
                        entity_base_name="",
                        item_name=enum.name,
                        item_base_name="",
                        data_type=0,
                        score=0.0,
                    ),
                )
            )

        return entries

    # ------------------------------------------------------------------
    # Embedder lazy init
    # ------------------------------------------------------------------

    def _ensure_embedder(self) -> OnnxEmbedder:
        if self._embedder is None:
            self._embedder = OnnxEmbedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Loading embeddings (with cache)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Ensure the embedding matrix is ready (load cache or encode)."""
        if self._embeddings is not None:
            return

        cached = self._cache.load(self._digest, len(self._corpus))
        if cached is not None:
            self._embeddings, self._corpus = cached
            return

        # No cache — encode the full corpus now.
        embedder = self._ensure_embedder()
        texts = [text for text, _ in self._corpus]
        log.info("Building semantic search index (%d entries) …", len(texts))
        emb = embedder.encode(texts, normalize=True)
        self._embeddings = emb
        self._cache.save(self._digest, emb, self._corpus)
        log.info("Search index built and cached.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Pre-load model weights and build the embedding matrix.

        Call this once on startup (e.g. in a background thread) so that the
        first :meth:`search` call has no latency.
        """
        self._ensure_loaded()
        # Ensure the embedder itself is ready for query encoding.
        self._ensure_embedder()._ensure_loaded()

    def search(self, query: str, top_k: int = 20) -> list[ModelMatch]:
        """Return up to *top_k* :class:`ModelMatch` objects sorted by cosine similarity.

        Returns an empty list for blank queries or empty corpora.
        """
        if not query.strip() or not self._corpus:
            return []

        self._ensure_loaded()
        assert self._embeddings is not None  # noqa: S101

        embedder = self._ensure_embedder()
        q_emb: npt.NDArray[np.float32] = embedder.encode([query], normalize=True)[0]

        scores: npt.NDArray[np.float32] = self._embeddings @ q_emb
        n = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results: list[ModelMatch] = []
        for i in top_idx:
            _, match = self._corpus[int(i)]
            results.append(
                ModelMatch(
                    kind=match.kind,
                    entity_name=match.entity_name,
                    entity_base_name=match.entity_base_name,
                    item_name=match.item_name,
                    item_base_name=match.item_base_name,
                    data_type=match.data_type,
                    score=float(scores[int(i)]),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Model introspection helpers
    # ------------------------------------------------------------------

    def entity_schema(self, entity_name: str) -> str:
        """Return a human-readable schema string for *entity_name*.

        Returns an empty string if the entity does not exist.  Useful for
        providing context to an LLM generating ODS query conditions.
        """
        if entity_name not in self._model.entities:
            return ""

        entity = self._model.entities[entity_name]
        lines: list[str] = [f"Entity: {entity_name} (base: {entity.base_name})"]

        if entity.attributes:
            lines.append("  Attributes:")
            for name, attr in entity.attributes.items():
                lines.append(f"    {name} (base: {attr.base_name}, type: {int(attr.data_type)})")

        if entity.relations:
            lines.append("  Relations:")
            for name, rel in entity.relations.items():
                lines.append(f"    {name} -> {rel.entity_name} (base: {rel.base_name})")

        return "\n".join(lines)

    def resolve_attribute(self, entity_name: str, attr_name: str) -> str | None:
        """Return the canonical attribute name for *attr_name* in *entity_name*.

        Performs an exact case-insensitive lookup first, then falls back to a
        semantic search if embeddings are loaded.

        Returns ``None`` if not found.
        """
        if entity_name not in self._model.entities:
            return None

        entity = self._model.entities[entity_name]
        lower = attr_name.lower()

        # Exact case-insensitive match.
        for name in entity.attributes:
            if name.lower() == lower:
                return str(name)

        # Semantic fallback (only if embeddings are already loaded).
        if self._embeddings is None:
            return None

        results = self.search(f"{entity_name} {attr_name}", top_k=5)
        entity_attrs = [
            r for r in results if r.entity_name == entity_name and r.kind == "attribute"
        ]
        if entity_attrs and entity_attrs[0].score > 0.45:
            return str(entity_attrs[0].item_name)

        return None

    def resolve_entity(self, name: str) -> str | None:
        """Return the canonical entity name for *name*, or ``None``.

        Only entities (not attributes or relations) are matched.  Performs a
        case-insensitive exact match.
        """
        lower = name.lower()
        for entity_name in self._model.entities:
            if entity_name.lower() == lower:
                return str(entity_name)
        return None

    def find_date_attribute(self, entity_name: str) -> str | None:
        """Return the primary date attribute of *entity_name*, or ``None``.

        When multiple DT_DATE attributes exist the one whose name contains
        *begin* or *start* is preferred, then *end*, then the first found.
        """
        if entity_name not in self._model.entities:
            return None

        entity = self._model.entities[entity_name]
        date_attrs: list[str] = [
            str(name) for name, attr in entity.attributes.items() if int(attr.data_type) == _DT_DATE
        ]

        if not date_attrs:
            return None
        if len(date_attrs) == 1:
            return date_attrs[0]

        for preference in ("begin", "start"):
            for a in date_attrs:
                if preference in a.lower():
                    return a
        for a in date_attrs:
            if "end" in a.lower():
                return a

        return date_attrs[0]

    # ------------------------------------------------------------------
    # Convenience / testing helpers
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> npt.NDArray[np.float32] | None:
        """The embedding matrix, or ``None`` if not yet loaded."""
        return self._embeddings

    def _inject_embeddings(
        self,
        embeddings: npt.NDArray[np.float32],
        corpus: list[tuple[str, ModelMatch]] | None = None,
    ) -> None:
        """Inject pre-built embeddings (used in tests to avoid real ONNX calls)."""
        self._embeddings = embeddings
        if corpus is not None:
            self._corpus = corpus

    @staticmethod
    def raises_unavailable(hint: str) -> SemanticSearchUnavailableError:
        """Factory for :class:`SemanticSearchUnavailableError`."""
        return SemanticSearchUnavailableError(hint)
