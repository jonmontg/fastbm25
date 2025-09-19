# FastBM25

A high-performance BM25 implementation in Rust with Python bindings.

## Installation

### Prerequisites

1. **Install Poetry**: https://python-poetry.org/docs/#installation
2. **Install Rust**: This project requires Rust to compile the core implementation

   **Install Rust:**
   ```bash
   # Install Rust using rustup (recommended)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   
   # Verify installation
   rustc --version
   cargo --version
   ```

   **Alternative installation methods:**
   - **macOS**: `brew install rust`
   - **Windows**: Download from https://rustup.rs/
   - **Linux**: Use your package manager or rustup

   **Why Rust is required:**
   - The core BM25 algorithm is implemented in Rust for performance
   - Maturin compiles the Rust code into a Python extension module
   - PyO3 creates the Python bindings during compilation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jonmontg/fastbm25.git
cd fastbm25

# Install dependencies and build
poetry install
poetry run maturin develop

# Run tests
poetry run python test_module.py
```

That's it! The package is now installed and ready to use.

## Usage

```python
import fastbm25

# Create a corpus (list of documents, where each document is tokenized).
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
# Install dependencies and build
poetry install
poetry run maturin develop

# Run tests
poetry run python test_module.py

# Format code
make format

# Run all checks
make check

# Build release wheel
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
