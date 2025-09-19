# FastBM25

A high-performance BM25 implementation in Rust with Python bindings.

## Features

- Fast BM25 ranking algorithm implementation
- Python bindings via PyO3
- Efficient memory usage
- Support for custom k1 and b parameters

## Installation

### Using Poetry (Recommended)

1. Install Poetry: https://python-poetry.org/docs/#installation
2. Install Rust: https://rustup.rs/
3. Install dependencies and build:
   ```bash
   poetry install
   ```

### Using pip

1. Install Rust: https://rustup.rs/
2. Install maturin: `pip install maturin`
3. Build and install:
   ```bash
   maturin develop
   ```

### Using the build script

```bash
python build.py
pip install target/wheels/fastbm25-*.whl
```

### Development with Poetry

```bash
# Install in development mode
poetry install

# Run tests
poetry run python test_module.py

# Build the package
poetry run maturin build --release

# Format code
make format

# Run all checks
make check
```

## Usage

```python
import fastbm25

# Create a corpus (list of documents, where each document is a list of words)
corpus = [
    ["hello", "world"],
    ["hello", "python"],
    ["python", "programming"]
]

# Initialize BM25
bm25 = fastbm25.BM25(corpus)

# Get scores for a query
query = ["hello", "python"]
scores = bm25.get_scores(query)
print(scores)  # [0.0, 1.0, 0.0]

# Get top-k document indices
top_indices = bm25.get_top_k_indices(query, k=2)
print(top_indices)  # [1, 0] (or similar, depending on scores)
```

## Parameters

- `k1`: Controls term frequency normalization (default: 1.5)
- `b`: Controls length normalization (default: 0.75)

## Development

### Using Poetry (Recommended)

```bash
# Install dependencies
poetry install

# Build in development mode
poetry run maturin develop

# Run tests
poetry run python test_module.py

# Format code
make format

# Run all checks
make check

# Build wheel
poetry run maturin build --release
```

### Using make commands

```bash
make help          # Show all available commands
make dev-install   # Install in development mode
make build         # Build the package
make test          # Run tests
make format        # Format code
make lint          # Run linting
make clean         # Clean build artifacts
```

### Using maturin directly

```bash
maturin develop
maturin build --release
```
