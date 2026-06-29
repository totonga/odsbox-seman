# Developer Guide

Welcome! This guide covers setup, testing, code standards, and the architecture for contributing to **odsbox-seman**.

## Quick Start

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (modern Python package manager)
- Git

### Clone & Setup

```bash
git clone https://github.com/totonga/odsbox-seman.git
cd odsbox-seman
uv sync --all-groups  # Install all dependencies (core + dev)
```

### Verify Installation

```bash
# Run all tests
uv run pytest tests/

# Run only unit tests (fast, no network)
uv run pytest tests/unit/ -v

# Run linting and type checking
uv run ruff check src/ tests/
uv run mypy src/

# Format code
uv run ruff format src/ tests/
```

## Project Structure

```
odsbox-seman/
├── src/odsbox_seman/           # Package source
│   ├── __init__.py             # Public API exports
│   ├── model_searcher.py       # Main ModelSearcher class (~350 lines)
│   ├── embedder.py             # ONNX embedding engine (~200 lines)
│   ├── cache.py                # Disk caching (NPZ + JSON) (~150 lines)
│   ├── tokenizer.py            # Name tokenization (~50 lines)
│   └── types.py                # ModelMatch, exceptions (~50 lines)
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── unit/                   # Unit tests (mocked, no network)
│   │   ├── test_tokenizer.py
│   │   ├── test_types.py
│   │   ├── test_cache.py
│   │   ├── test_embedder.py
│   │   ├── test_searcher.py
│   │   └── test_errors.py
│   ├── integration/            # Integration tests (real ONNX model)
│   │   └── test_search_integration.py
│   └── data/
│       └── mdm_nvh_model.json  # Test fixture (ODS model protobuf)
├── .github/
│   ├── workflows/ci.yml        # GitHub Actions CI/CD
│   └── dependabot.yml          # Automated dependency updates
├── pyproject.toml              # Project config + dependencies
├── README.md                   # User guide
├── DEVELOPER.md                # This file
├── LICENSE                     # Apache 2.0
└── uv.lock                     # Lockfile (versioned)
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

Edit files in `src/odsbox_seman/` and/or `tests/`.

### 3. Run Tests Locally

**Fast unit tests (no network):**

```bash
uv run pytest tests/unit/ -v --cov=src --cov-fail-under=85
```

Must pass all tests and maintain ≥85% code coverage.

**Full test suite (with integration tests, ~3 min):**

```bash
uv run pytest tests/ -v --cov=src --cov-report=html
# Open htmlcov/index.html to inspect coverage
```

### 4. Lint & Format

```bash
uv run ruff check src/ tests/ --fix  # Auto-fix lint issues
uv run ruff format src/ tests/       # Auto-format code
uv run mypy src/                      # Type check (strict mode)
```

### 5. Commit & Push

```bash
git add src/ tests/ pyproject.toml
git commit -m "feat: add my feature"
git push origin feature/my-feature
```

### 6. Open a Pull Request

Push to GitHub and open a PR. CI will run automatically.

## Code Standards

### Style & Formatting

- **Line length**: 100 characters (enforced by Ruff)
- **Imports**: Sorted alphabetically, grouped (stdlib → third-party → local)
- **Type hints**: Required on all functions (mypy strict mode)
- **Docstrings**: Google-style for modules, classes, and public methods

Example:

```python
def search(self, query: str, top_k: int = 20) -> list[ModelMatch]:
    """Return up to *top_k* results sorted by cosine similarity.

    Parameters
    ----------
    query:
        Natural language search string.
    top_k:
        Maximum number of results to return. Default: 20.

    Returns
    -------
    list[ModelMatch]
        Results ranked by cosine similarity (descending).
    """
```

### Linting Rules

Enforced by Ruff:
- **E** (pycodestyle errors)
- **F** (Pyflakes)
- **I** (isort imports)
- **UP** (pyupgrade modernization)
- **B** (flake8-bugbear)
- **SIM** (flake8-simplify)

Run `uv run ruff check src/ tests/` to see violations.

### Type Checking

Mypy in strict mode ensures full type coverage:

```bash
uv run mypy src/
```

No untyped functions or `Any` types allowed (except where explicitly necessary with `# type: ignore`).

## Testing

### Unit Tests

Located in `tests/unit/`. These use mocks and avoid network calls.

```bash
uv run pytest tests/unit/ -v
```

**Coverage target: ≥85%**

Test structure:

```python
class TestMyFeature:
    def test_basic_case(self) -> None:
        # Arrange
        obj = MyClass()
        
        # Act
        result = obj.method()
        
        # Assert
        assert result == expected
```

### Integration Tests

Located in `tests/integration/`. These download the real ONNX model and encode actual texts.

```bash
uv run pytest tests/integration/ -v -m integration
```

Marked with `@pytest.mark.integration` so they can be skipped in CI if needed.

### Coverage Reports

```bash
# Generate HTML coverage report
uv run pytest tests/unit/ --cov=src --cov-report=html
open htmlcov/index.html
```

Coverage is enforced at 85% minimum; the CI/CD pipeline will fail if coverage drops below this.

## Architecture

### Data Flow

```
ODS Model (protobuf)
    ↓
ModelSearcher
    ├─ Corpus Building
    │   └─ _build_corpus()
    │       ├─ Extract entities → tokenize name → ModelMatch
    │       ├─ Extract attributes → tokenize name → ModelMatch
    │       ├─ Extract relations → tokenize name → ModelMatch
    │       └─ Extract enumerations → tokenize name → ModelMatch
    │
    ├─ Cache Layer (CacheManager)
    │   ├─ Compute SHA-256 hash of model
    │   ├─ Check ~/.ods-seman/search_cache/{hash}.npz
    │   └─ If found: load embeddings; else: encode corpus
    │
    ├─ ONNX Embedder (OnnxEmbedder)
    │   ├─ Lazy load: download model from HF Hub (first use only)
    │   ├─ Auto-detect provider: CUDA → DirectML → CPU
    │   ├─ Batch encode: split corpus into chunks of 64
    │   ├─ Tokenize with transformers.AutoTokenizer
    │   ├─ Run ONNX inference
    │   └─ Mean-pool + L2-normalize embeddings
    │
    └─ Search (search())
        ├─ Encode query (same embedder)
        ├─ Compute dot product (corpus @ query)
        ├─ Sort by score (descending)
        └─ Return top_k ModelMatch objects
```

### Key Classes

**`ModelSearcher`** (model_searcher.py)
- Main public interface
- Owns: model protobuf, corpus, embeddings, embedder, cache manager
- Public methods: `search()`, `warm_up()`, `entity_schema()`, `resolve_entity()`, `resolve_attribute()`, `find_date_attribute()`

**`OnnxEmbedder`** (embedder.py)
- Encapsulates ONNX Runtime inference
- Lazy loads model from HF Hub
- Auto-detects GPU provider
- Handles batch encoding with chunking

**`CacheManager`** (cache.py)
- Manages embedding disk cache
- Key: SHA-256 hash of model protobuf
- Stores: `{hash}.npz` (embeddings) + `{hash}.json` (corpus metadata)

**`ModelMatch`** (types.py)
- Dataclass representing a search result
- Fields: kind, entity_name, item_name, score, etc.

### Tokenization

The `tokenize()` function (tokenizer.py) handles:
- CamelCase splitting (`MeaQuantity` → `mea quantity`)
- snake_case splitting (`tyre_pressure` → `tyre pressure`)
- Hyphen splitting (`some-attr` → `some attr`)
- Lowercasing and whitespace normalization

This ensures that entity/attribute names are properly split for the embedding model.

## Dependency Management

### Adding Dependencies

Add to core dependencies:

```bash
uv add package-name
```

Add to dev dependencies:

```bash
uv add --dev package-name
```

The `pyproject.toml` and `uv.lock` will be updated automatically.

### Security Audits

```bash
uv run pip-audit --strict
```

Runs in CI on every push.

## Release & Versioning

### Semantic Versioning

Releases follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Bumped automatically by `python-semantic-release` based on commit messages

### Commit Message Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: fix a bug
docs: update documentation
test: add tests
chore(deps): update dependencies
```

Examples:

```
feat: add batch encoding to OnnxEmbedder
fix: handle missing token_type_ids from tokenizer
docs: update README with examples
test: add test for cache corruption recovery
chore(deps): update transformers to 4.36
```

CI automatically bumps the version and publishes to PyPI when these commits land on `main`.

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

searcher = ModelSearcher(model)
searcher.warm_up()  # Will now print debug logs
```

Logs from odsbox-seman include:
- ONNX model download progress
- Provider selection (CPU/CUDA/DirectML)
- Cache hits/misses
- Corpus building steps

### Inspect Embeddings

```python
searcher.warm_up()
emb = searcher.embeddings  # NumPy array of shape (n_corpus, embedding_dim)
print(emb.shape)
print(emb[0])  # First embedding (should be normalized: norm ≈ 1.0)
```

### Test with Small Model

The test fixture `mdm_nvh_model.json` is a small ODS model (≈50 entities) used in tests. Use it for quick debugging:

```python
from google.protobuf.json_format import Parse
from odsbox.proto.ods_pb2 import Model
from odsbox_seman import ModelSearcher

model = Model()
with open("tests/data/mdm_nvh_model.json") as f:
    Parse(f.read(), model)

searcher = ModelSearcher(model)
results = searcher.search("pressure")
for r in results:
    print(r)
```

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to `main` and every PR:

1. **Lint & Type Check** (Ubuntu)
   - `ruff check` and `ruff format --check`
   - `mypy src/`

2. **Unit Tests** (Ubuntu)
   - `pytest tests/unit/` with 85% coverage gate
   - Fails if coverage drops below 85%

3. **Integration Tests** (Ubuntu)
   - `pytest tests/integration/ -m integration`
   - Real ONNX model download and encoding

4. **Dependency Audit** (Ubuntu)
   - `pip-audit --strict`
   - Detects known vulnerabilities

5. **Release** (Ubuntu, main branch only)
   - Auto-version with `python-semantic-release`
   - Build wheel with `uv build`
   - Publish to PyPI with trusted publishing

### Dependabot

Automated weekly dependency updates are configured in `.github/dependabot.yml`:
- Python packages: up to 10 PRs/week
- GitHub Actions: up to 5 PRs/week

## Common Tasks

### Add a New Public Method to ModelSearcher

1. Implement in `src/odsbox_seman/model_searcher.py`
2. Add comprehensive docstring (Google-style)
3. Add unit tests in `tests/unit/test_searcher.py`
4. Update the public `__all__` in `src/odsbox_seman/__init__.py` if needed
5. Run `uv run pytest tests/unit/ -k "your_feature"` to test
6. Run `uv run mypy src/` to ensure type safety

### Fix a Bug

1. Write a test that reproduces the bug in `tests/unit/`
2. Implement the fix in `src/`
3. Verify the test now passes
4. Run full test suite: `uv run pytest tests/unit/ --cov=src --cov-fail-under=85`
5. Commit with message: `fix: description of fix`

### Update Documentation

1. Update `README.md` for user-facing changes
2. Update `DEVELOPER.md` for developer-facing changes
3. Ensure docstrings in code are synchronized
4. Test that code examples actually work

## Troubleshooting Development Issues

### "mypy found 5 errors"

Run `uv run mypy src/` to see the errors. Most common issues:
- Missing type hint on function parameter/return
- Using `Any` where specific type is known

Fix with proper type annotations.

### "Coverage dropped to 84%"

Run `uv run pytest tests/unit/ --cov=src --cov-report=html` and open `htmlcov/index.html` to see which lines are uncovered. Add test cases for those lines.

### "pytest hangs on integration tests"

Integration tests download the ONNX model (~50 MB) from HF Hub on first run. This can take 1-2 minutes on slow networks. Subsequent runs will use the cached model.

To skip integration tests:

```bash
uv run pytest tests/unit/  # Only unit tests
```

### "onnxruntime.capi.onnx_runtime_c_api._assert_c_api_status: ONNX Runtime error"

Usually caused by:
1. Mismatch between tokenizer output and model input expectations
2. Batch size too large (OOM)

Check:
- Is `token_type_ids` required by the model but missing from tokenizer output?
- Are sequences truncated to `_MAX_LENGTH` (512)?
- Does batch size exceed available GPU/CPU memory?

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions about usage or development
- **Email**: totonga@gmail.com

## License

All contributions are under the Apache License 2.0 (see [LICENSE](LICENSE)).
