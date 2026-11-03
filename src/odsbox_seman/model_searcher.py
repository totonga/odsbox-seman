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
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from odsbox_seman.cache import CacheManager, model_hash
from odsbox_seman.config import DEFAULT_ALIAS_BOOST, _matches_any
from odsbox_seman.embedder import OnnxEmbedder
from odsbox_seman.tokenizer import tokenize
from odsbox_seman.types import (
    CorpusStats,
    ModelMatch,
    MultiSearchResult,
    SearchMode,
    SearchSection,
    SemanticSearchUnavailableError,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from odsbox.proto.ods_pb2 import Model

    from odsbox_seman.config import ModelConfig

log = logging.getLogger(__name__)
progress_log = logging.getLogger(f"{__name__}.progress")

# DataTypeEnum value for DT_DATE in the ODS protobuf spec.
_DT_DATE = 10


# Alias boost ramps up from 0 at this raw similarity to full at _BOOST_FULL.
_BOOST_FLOOR = 0.3
_BOOST_FULL = 0.6


def _casefold_keys[V](mapping: dict[str, V]) -> dict[str, V]:
    return {k.lower(): v for k, v in mapping.items()}


def _lookup(folded: dict[str, tuple[str, ...]], *names: str) -> tuple[str, ...]:
    """Return the aliases stored under any of the case-insensitive names (union)."""
    found: list[str] = []
    for name in dict.fromkeys(n.lower() for n in names if n):
        found.extend(folded.get(name, ()))
    return tuple(dict.fromkeys(found))


def _lookup_nested(
    folded: dict[str, dict[str, tuple[str, ...]]], *names: str
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {}
    for name in dict.fromkeys(n.lower() for n in names if n):
        for key, aliases in folded.get(name, {}).items():
            merged[key.lower()] = tuple(dict.fromkeys((*merged.get(key.lower(), ()), *aliases)))
    return merged


class ModelSearcher:
    """Semantic search index built from an ASAM ODS :class:`~odsbox.proto.ods_pb2.Model`.

    Concurrent initialization waits for the first cache build for the same cache
    directory and model/config digest within this process. Failed builds can be retried.

    Parameters
    ----------
    model:
        Parsed ODS model protobuf.
    cache_dir:
        Custom cache directory. If omitted, uses ``ODSBOX_SEMAN_CACHE_DIR`` when set;
        otherwise defaults to ``~/.ods-seman/search_cache/``.
    config:
        Optional per-model settings (blacklists and aliases).
    """

    def __init__(
        self,
        model: Model,
        cache_dir: Path | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._alias_boost: float = config.alias_boost if config else DEFAULT_ALIAS_BOOST
        self._digest: str = model_hash(model, config)
        self._cache = CacheManager(cache_dir)
        self._alias_flags: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._stats = CorpusStats()
        self._corpus: list[tuple[str, ModelMatch]] = self._build_corpus()
        self._embeddings: npt.NDArray[np.float32] | None = None
        self._embedder: OnnxEmbedder | None = None
        self._embedder_lock = Lock()
        self._warn_unmatched_config()

    @property
    def stats(self) -> CorpusStats:
        """Counts of kept versus blacklisted model elements."""
        return self._stats

    # ------------------------------------------------------------------
    # Corpus construction
    # ------------------------------------------------------------------

    def _build_corpus(self) -> list[tuple[str, ModelMatch]]:
        """Build the ``(text, ModelMatch)`` pairs from the ODS model and the config."""
        cfg = self._config
        entries: list[tuple[str, ModelMatch]] = []
        flags: list[float] = []

        def add(text: str, match: ModelMatch, *, alias: bool = False) -> None:
            entries.append((text, match))
            flags.append(1.0 if alias else 0.0)

        entity_aliases = _casefold_keys(cfg.aliases.entities) if cfg else {}
        attr_aliases = _casefold_keys(cfg.aliases.attributes) if cfg else {}
        enum_aliases = _casefold_keys(cfg.aliases.enumerations) if cfg else {}

        entities_total = attributes_total = relations_total = 0
        attributes_kept = relations_kept = 0
        excluded_entities: list[str] = []
        excluded_attributes: list[str] = []

        for entity in self._model.entities.values():
            entities_total += 1
            attributes_total += len(entity.attributes)
            relations_total += len(entity.relations)
            if cfg and cfg.is_entity_blacklisted(entity.name, entity.base_name):
                log.debug("Blacklisted entity %s", entity.name)
                excluded_entities.append(entity.name)
                continue

            aliases = _lookup(entity_aliases, entity.name, entity.base_name)
            e_text = f"{tokenize(entity.name)} {tokenize(entity.base_name)}"
            # Entity aliases also steer attribute and relation matches of that entity.
            e_text_full = " ".join([e_text, *(tokenize(a) for a in aliases)])

            entity_match = ModelMatch(
                kind="entity",
                entity_name=entity.name,
                entity_base_name=entity.base_name,
                item_name=entity.name,
                item_base_name=entity.base_name,
                data_type=0,
                score=0.0,
            )
            add(e_text, entity_match)
            for alias in aliases:
                # Bare alias text: adding the entity name dilutes the match for the alias itself.
                add(tokenize(alias), entity_match, alias=True)
            if len(aliases) > 1:
                # Related aliases together give the model more context than each one alone.
                add(" ".join(tokenize(a) for a in aliases), entity_match, alias=True)

            entity_attr_aliases = _casefold_keys(
                _lookup_nested(attr_aliases, entity.name, entity.base_name)
            )

            for attr in entity.attributes.values():
                if cfg and cfg.is_attribute_blacklisted(attr.name, attr.base_name):
                    log.debug("Blacklisted attribute %s.%s", entity.name, attr.name)
                    excluded_attributes.append(f"{entity.name}.{attr.name}")
                    continue
                attributes_kept += 1
                a_text = f"{tokenize(attr.name)} {tokenize(attr.base_name)}"
                attr_match = ModelMatch(
                    kind="attribute",
                    entity_name=entity.name,
                    entity_base_name=entity.base_name,
                    item_name=attr.name,
                    item_base_name=attr.base_name,
                    data_type=int(attr.data_type),
                    score=0.0,
                )
                add(f"{e_text_full} {a_text}", attr_match)
                for alias in _lookup(entity_attr_aliases, attr.name, attr.base_name):
                    add(tokenize(alias), attr_match, alias=True)

            for rel in entity.relations.values():
                if cfg and (
                    cfg.is_attribute_blacklisted(rel.name, rel.base_name)
                    or cfg.is_entity_blacklisted(rel.entity_name, rel.entity_base_name)
                ):
                    log.debug("Blacklisted relation %s.%s", entity.name, rel.name)
                    continue
                relations_kept += 1
                text = (
                    f"{e_text_full} {tokenize(rel.name)} {tokenize(rel.base_name)} "
                    f"{tokenize(rel.entity_name)} {tokenize(rel.entity_base_name)} "
                    f"{tokenize(rel.inverse_name)} {tokenize(rel.inverse_base_name)}"
                )
                add(
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

        for enum in self._model.enumerations.values():
            items_text = " ".join(tokenize(k) for k in enum.items)
            add(
                f"{tokenize(enum.name)} {items_text}",
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
            for item, item_aliases in _casefold_keys(
                _lookup_nested(enum_aliases, enum.name)
            ).items():
                original = next((k for k in enum.items if k.lower() == item), item)
                for alias in item_aliases:
                    add(
                        tokenize(alias),
                        ModelMatch(
                            kind="enumeration",
                            entity_name="",
                            entity_base_name="",
                            item_name=enum.name,
                            item_base_name=original,
                            data_type=0,
                            score=0.0,
                        ),
                        alias=True,
                    )

        self._alias_flags = np.asarray(flags, dtype=np.float32)
        self._stats = CorpusStats(
            entities_total=entities_total,
            entities_kept=entities_total - len(excluded_entities),
            attributes_total=attributes_total,
            attributes_kept=attributes_kept,
            relations_total=relations_total,
            relations_kept=relations_kept,
            enumerations=len(self._model.enumerations),
            excluded_entities=tuple(excluded_entities),
            excluded_attributes=tuple(excluded_attributes),
        )
        log.info(
            "Corpus: %d/%d entities, %d/%d attributes, %d/%d relations, %d enumerations "
            "(%d entries, %d aliases).",
            self._stats.entities_kept,
            entities_total,
            attributes_kept,
            attributes_total,
            relations_kept,
            relations_total,
            self._stats.enumerations,
            len(entries),
            int(sum(flags)),
        )
        return entries

    def _warn_unmatched_config(self) -> None:
        """Report blacklist patterns and alias keys that match nothing in the model."""
        cfg = self._config
        if cfg is None:
            return
        entities = list(self._model.entities.values())
        attrs = [a for e in entities for a in e.attributes.values()]
        rels = [r for e in entities for r in e.relations.values()]
        for pattern in cfg.blacklist.entities:
            if not any(_matches_any((pattern,), e.name, e.base_name) for e in entities):
                log.warning("Model '%s': entity blacklist '%s' matches nothing.", cfg.name, pattern)
        for pattern in cfg.blacklist.attributes:
            if not any(_matches_any((pattern,), x.name, x.base_name) for x in (*attrs, *rels)):
                log.warning(
                    "Model '%s': attribute blacklist '%s' matches nothing.", cfg.name, pattern
                )
        # Global aliases are shared across models, so unmatched keys are expected; keep INFO.
        known = {n.lower() for e in entities for n in (e.name, e.base_name) if n}
        for key in (*cfg.aliases.entities, *cfg.aliases.attributes):
            if key.lower() not in known:
                log.info("Model '%s': alias key '%s' matches no entity.", cfg.name, key)

    # ------------------------------------------------------------------
    # Embedder lazy init
    # ------------------------------------------------------------------

    def _ensure_embedder(self) -> OnnxEmbedder:
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = OnnxEmbedder()
            return self._embedder

    # ------------------------------------------------------------------
    # Loading embeddings (with cache)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Ensure the embedding matrix is ready (load cache or encode)."""
        with self._cache.lock(self._digest):
            if self._embeddings is not None:
                return

            cached = self._cache.load(self._digest, len(self._corpus))
            if cached is not None:
                embeddings, self._corpus, self._alias_flags = cached
                self._embeddings = embeddings
                return

            progress_log.info(
                "No cached search index available. Tokenizing and embedding model (%d entries); "
                "this may take a while ...",
                len(self._corpus),
            )
            embedder = self._ensure_embedder()
            texts = [text for text, _ in self._corpus]
            emb = embedder.encode(texts, normalize=True)
            self._cache.save(self._digest, emb, self._corpus, self._alias_flags)
            self._embeddings = emb
            progress_log.info("Search index built and cached.")

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

        return self._search_ranked(query, top_k)

    def search_multi(
        self, query: str, top_k: int = 20, *, entity_name: str | None = None
    ) -> MultiSearchResult:
        """Search every mode once and return per-mode top-k and their ranked union.

        Entity filtering is a case-insensitive substring match. Timings exclude
        initialization; section timings cover selection, not shared scoring.
        Blank queries, empty corpora and nonpositive limits produce empty sections.
        """
        modes: tuple[SearchMode, ...] = ("entity", "attribute", "relation", "enumeration", "all")
        if not query.strip() or not self._corpus or top_k <= 0:
            return MultiSearchResult(
                [SearchSection(mode, [], 0.0) for mode in modes], [], 0.0, 0.0, 0.0
            )

        self.warm_up()
        started = perf_counter()
        ranked = self._search_ranked(query, None)
        shared_seconds = perf_counter() - started
        sections: list[SearchSection] = []
        for mode in modes:
            selection_started = perf_counter()
            matches: list[ModelMatch] = []
            for match in ranked:
                if mode != "all" and match.kind != mode:
                    continue
                if entity_name is not None and entity_name.lower() not in match.entity_name.lower():
                    continue
                matches.append(match)
                if len(matches) >= top_k:
                    break
            sections.append(SearchSection(mode, matches, perf_counter() - selection_started))

        merge_started = perf_counter()
        selected = {
            (match.kind, match.entity_name, match.item_name, match.item_base_name)
            for section in sections
            for match in section.matches
        }
        merged = [
            match
            for match in ranked
            if (match.kind, match.entity_name, match.item_name, match.item_base_name) in selected
        ]
        merge_seconds = perf_counter() - merge_started
        return MultiSearchResult(
            sections, merged, shared_seconds, merge_seconds, perf_counter() - started
        )

    def _search_ranked(self, query: str, top_k: int | None) -> list[ModelMatch]:
        self._ensure_loaded()
        assert self._embeddings is not None  # noqa: S101

        embedder = self._ensure_embedder()
        q_emb: npt.NDArray[np.float32] = embedder.encode([query], normalize=True)[0]

        raw: npt.NDArray[np.float32] = self._embeddings @ q_emb
        # Weak matches must not profit from the alias boost, or unrelated aliases outrank real hits.
        gate = np.clip((raw - _BOOST_FLOOR) / (_BOOST_FULL - _BOOST_FLOOR), 0.0, 1.0)
        scores: npt.NDArray[np.float32] = np.minimum(
            raw + self._alias_flags * self._alias_boost * gate, 1.0
        ).astype(np.float32)

        # Walk in score order and keep the best entry per item (alias entries duplicate items).
        results: list[ModelMatch] = []
        seen: set[tuple[str, str, str, str]] = set()
        for i in np.argsort(-scores):
            _, match = self._corpus[int(i)]
            key = (match.kind, match.entity_name, match.item_name, match.item_base_name)
            if key in seen:
                continue
            seen.add(key)
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
            if top_k is not None and len(results) >= top_k:
                break
        log.debug("Search %r: %s", query, [(r.item_name, round(r.score, 3)) for r in results[:5]])
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
        alias_flags: npt.NDArray[np.float32] | None = None,
    ) -> None:
        """Inject pre-built embeddings (used in tests to avoid real ONNX calls)."""
        self._embeddings = embeddings
        if corpus is not None:
            self._corpus = corpus
            self._alias_flags = np.zeros(len(corpus), dtype=np.float32)
        if alias_flags is not None:
            self._alias_flags = alias_flags

    @staticmethod
    def raises_unavailable(hint: str) -> SemanticSearchUnavailableError:
        """Factory for :class:`SemanticSearchUnavailableError`."""
        return SemanticSearchUnavailableError(hint)
