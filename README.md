# odsbox-seman

Lightweight semantic search for ASAM ODS models using ONNX embeddings.

[![CI](https://github.com/totonga/odsbox-seman/actions/workflows/ci.yml/badge.svg)](https://github.com/totonga/odsbox-seman/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/odsbox-seman.svg)](https://pypi.org/project/odsbox-seman/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

**odsbox-seman** is a Python package that enables semantic search on ASAM ODS (Open Data Standard) model definitions. It uses **ONNX Runtime** with the multilingual MiniLM sentence embedding model to index and search ODS entities, attributes, relations, and enumerations by meaning rather than exact string matching.

### Key Features

- 🚀 **PyTorch-free** — Uses ONNX Runtime (~15 MB) instead of PyTorch (~500 MB+)
- 🌍 **Multilingual** — Supports 100+ languages via paraphrase-multilingual-MiniLM-L12-v2
- 💾 **Disk cached** — Embeddings cached locally; model downloaded on first use from Hugging Face Hub
- 🔄 **Auto GPU** — Detects and uses CUDA or DirectML when available; gracefully falls back to CPU
- 📦 **Production-ready** — 92% test coverage, full type hints, mypy strict mode

## Installation

### Basic (CPU)

```bash
pip install odsbox-seman
```

or with `uv`:

```bash
uv add odsbox-seman
```

### GPU Support (CUDA on Linux/Windows)

```bash
pip install odsbox-seman[gpu]
```

This installs `onnxruntime-gpu` for NVIDIA GPU acceleration.

### DirectML Support (Windows Copilot+ NPU/GPU)

DirectML (Windows native GPU) support is available via the `npu` extra:

```bash
pip install odsbox-seman[npu]
```

The package will auto-detect and use DirectML if available.

## Quick Start

### 1. Load an ODS Model

```python
from pathlib import Path
from google.protobuf.json_format import Parse
from odsbox.proto.ods_pb2 import Model

# Load your ASAM ODS model from a JSON protobuf file
model = Model()
with open("mdm_nvh_model.json") as f:
    Parse(f.read(), model)
```

### 2. Create a Searcher

```python
from odsbox_seman import ModelSearcher

searcher = ModelSearcher(model)
# On first use, the ONNX model downloads from HF Hub (~50 MB, cached thereafter)
searcher.warm_up()  # Optional: pre-load embeddings for zero-latency first search
```

### 3. Search

```python
# Semantic search with natural language
results = searcher.search("tyre pressure sensor", top_k=10)

for match in results:
    print(f"{match.entity_name}.{match.item_name}  (score: {match.score:.3f})")
    # Output:
    # Wheel.Pressure  (score: 0.847)
    # TestEquipment.Pressure  (score: 0.823)
    # ...
```

## Usage Examples

For CLI usage and interactive examples, see [docs/CLI.md](docs/CLI.md).

### Search by Concept

Find all attributes related to measurement date:

```python
results = searcher.search("measurement date time", top_k=5)
for r in results:
    print(r)
```

### Entity Schema Introspection

Get the attribute list for an entity:

```python
schema = searcher.entity_schema("MeaResult")
print(schema)
# Entity: MeaResult (base: AoMeasurement)
#   Attributes:
#     Name (base: ao_name, type: 1)
#     CreationTime (base: ao_creation_time, type: 10)
#     ...
```

### Resolve Entity / Attribute Names

Find canonical names (case-insensitive):

```python
# Entity resolution
canonical_entity = searcher.resolve_entity("mearesult")  # → "MeaResult"

# Attribute resolution (with semantic fallback)
canonical_attr = searcher.resolve_attribute("MeaResult", "name")  # → "Name"
canonical_attr = searcher.resolve_attribute("MeaResult", "creation time")  # → "CreationTime"
```

### Find Date Attributes

Locate the primary date column for time-based filtering:

```python
date_attr = searcher.find_date_attribute("MeaResult")
print(date_attr)  # → "CreationTime"
```

### Custom Cache Directory

By default, embeddings are cached in `~/.ods-seman/search_cache/`. Override it:

```python
from pathlib import Path

searcher = ModelSearcher(model, cache_dir=Path("/custom/cache/dir"))
```

## Performance

- **First search** (with `warm_up()`): ~2-3 seconds (model loads, corpus encoded)
- **Subsequent searches**: ~50-100 ms (cached embeddings, query encoded)
- **Memory**: ~200-500 MB for models with 1000+ entities (ONNX embeddings cached in RAM)

## API Reference

### `ModelSearcher`

Main class for semantic search over an ODS model.

**Constructor:**

```python
ModelSearcher(model: Model, cache_dir: Path | None = None) -> None
```

**Methods:**

- `search(query: str, top_k: int = 20) -> list[ModelMatch]`  
  Return up to `top_k` results sorted by cosine similarity (descending).

- `warm_up() -> None`  
  Pre-load the ONNX model and build the embedding matrix. Call once on startup.

- `entity_schema(entity_name: str) -> str`  
  Return a human-readable schema string for debugging / LLM context.

- `resolve_entity(name: str) -> str | None`  
  Case-insensitive entity name lookup.

- `resolve_attribute(entity_name: str, attr_name: str) -> str | None`  
  Case-insensitive attribute lookup with semantic fallback.

- `find_date_attribute(entity_name: str) -> str | None`  
  Return the primary date attribute (prefers "begin"/"start", then "end").

**Properties:**

- `embeddings: NDArray[float32] | None`  
  The embedding matrix (shape `(n_corpus, embedding_dim)`), or `None` before loading.

### `ModelMatch`

Result of a search query.

**Fields:**

- `kind: Literal["attribute", "relation", "enumeration"]`
- `entity_name: str` — Name of the entity (empty for enumerations)
- `entity_base_name: str` — Base name (AoXyz) of the entity
- `item_name: str` — Attribute, relation, or enumeration name
- `item_base_name: str` — Base name of the item
- `data_type: int` — ODS DataTypeEnum value (0 for relations/enumerations)
- `score: float` — Cosine similarity in [0, 1]

### `SemanticSearchUnavailableError`

Raised when a required dependency is missing (e.g., `onnxruntime`, `transformers`).

## Troubleshooting

### "Semantic search requires 'onnxruntime'"

Install ONNX Runtime:

```bash
pip install onnxruntime
# or with GPU:
pip install onnxruntime-gpu
```

### "Semantic search requires 'transformers'"

Install the HuggingFace transformers library:

```bash
pip install transformers
```

### "Semantic search requires 'huggingface-hub'"

Install the HuggingFace Hub client:

```bash
pip install huggingface-hub
```

### Model download hangs or times out

The ONNX model (~50 MB) downloads from Hugging Face Hub on first use. If your network is slow or unstable, manually download it:

```python
from huggingface_hub import snapshot_download

# Pre-download to cache
cache_dir = snapshot_download("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(f"Cached to: {cache_dir}")
```

### GPU not detected

Check that your GPU provider is installed:

```python
import onnxruntime as ort

print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider', 'DmlExecutionProvider', etc.
```

If absent, install the appropriate package:

- **CUDA**: `pip install onnxruntime-gpu`
- **DirectML (Windows)**: `pip install onnxruntime-directml`

## Architecture

```
ModelSearcher
├─ Corpus Building
│  └─ Extract all entities, attributes, relations, enumerations
│     └─ Tokenize names (split snake_case, CamelCase)
├─ Caching Layer
│  └─ SHA-256 hash of model protobuf → {hash}.npz (embeddings) + {hash}.json (metadata)
├─ ONNX Embedder
│  ├─ Lazy load multilingual-MiniLM model from HF Hub
│  ├─ Auto-detect GPU provider (CUDA → DirectML → CPU)
│  ├─ Batch encode texts in chunks of 64 to avoid OOM
│  └─ L2-normalize embeddings for fast cosine similarity
└─ Search
   └─ Encode query → dot product with corpus embeddings → top-k ranking
```

## Development

See [docs/DEVELOPER.md](docs/DEVELOPER.md) for setup, testing, and contribution guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Citation

If you use odsbox-seman in research or production, please cite:

```bibtex
@software{odsbox_seman_2026,
  title={odsbox-seman: Semantic Search for ASAM ODS Models},
  author={totonga},
  year={2026},
  url={https://github.com/totonga/odsbox-seman}
}
```

## Related Projects

- [odsbox](https://github.com/totonga/odsbox) — Core ODS protobuf handling
- [odsbox-pilot](https://github.com/totonga/odsbox-pilot) — Desktop query tool
- [sentence-transformers](https://www.sbert.net/) — Semantic embeddings (reference)

## Contributing

Contributions welcome! Please see [docs/DEVELOPER.md](docs/DEVELOPER.md) for code standards and testing requirements.
