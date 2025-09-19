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

The library expects words to be encoded as unique integers to the time required to copy Python objects to Rust.
Using integer tokens reduces the time required to build a large IDF index by roughly 100x.

```python
import fastbm25

# Step 1: Prepare your text data
text_corpus = [
    ["hello", "world"],
    ["hello", "python"],
    ["python", "programming", "language"],
    ["rust", "programming", "language"],
    ["hello", "rust", "world"]
]

# Step 2: Create a vocabulary mapping (text tokens to numeric IDs)
tokens = list(set([token for doc in text_corpus for token in doc]))
token_to_id = {token: token_id for token_id, token in enumerate(tokens)}
id_to_token = {token_id: token for token, token_id in token_to_id.items()}

# Step 3: Convert text corpus to numeric IDs
corpus = [[token_to_id[token] for token in doc] for doc in text_corpus]

# Step 4: Initialize BM25 with numeric corpus
bm25 = fastbm25.BM25(corpus)

# Step 5: Prepare query (convert text to numeric IDs)
query_text = ["hello", "python"]
query = [token_to_id[token] for token in query_text]

# Step 6: Get BM25 scores for all documents
scores = bm25.get_scores(query)
print(f"Scores: {scores}")
# Output: [0.151, 0.488, 0.336, 0.0, 0.151]

# Step 7: Get top-k most relevant document indices
top_indices = bm25.get_top_k_indices(query, k=3)
print(f"Top 3 documents: {top_indices}")
# Output: [1, 2, 0] (indices of most relevant documents)

# Step 8: Get the actual text of top documents
for i, doc_idx in enumerate(top_indices):
    doc_text = [id_to_token[token_id] for token_id in corpus[doc_idx]]
    print(f"Rank {i+1}: {doc_text} (score: {scores[doc_idx]:.3f})")
```

### Helper Function for Easier Usage

```python
def create_bm25_from_texts(texts):
    """Helper function to create BM25 from list of text documents."""
    # Tokenize (simple whitespace splitting - you may want more sophisticated tokenization)
    corpus = [text.split() for text in texts]
    
    # Create vocabulary
    tokens = list(set([token for doc in corpus for token in doc]))
    token_to_id = {token: token_id for token_id, token in enumerate(tokens)}
    
    # Convert to numeric IDs
    numeric_corpus = [[token_to_id[token] for token in doc] for doc in corpus]
    
    return fastbm25.BM25(numeric_corpus), token_to_id

# Usage with helper function
texts = [
    "hello world",
    "hello python", 
    "python programming language",
    "rust programming language",
    "hello rust world"
]

bm25, token_to_id = create_bm25_from_texts(texts)

# Query
query_text = "hello python".split()
query = [token_to_id[token] for token in query_text if token in token_to_id]

scores = bm25.get_scores(query)
top_docs = bm25.get_top_k_indices(query, k=2)
```

## Parameters

The BM25 algorithm uses two main parameters that can be tuned for your specific use case:

- **`k1`**: Term frequency saturation parameter (default: 1.5)
  - Controls how quickly term frequency saturates
  - Higher values (1.5-2.0) give more weight to term frequency
  - Lower values (0.5-1.2) reduce the impact of term frequency
  - Typical range: 1.2 to 2.0

- **`b`**: Length normalization parameter (default: 0.75)
  - Controls how much document length affects scoring
  - 0.0 = no length normalization (longer docs not penalized)
  - 1.0 = full length normalization (longer docs heavily penalized)
  - 0.75 = balanced normalization (recommended)
  - Typical range: 0.0 to 1.0

### Parameter Tuning Tips

- **For short documents** (tweets, titles): Use higher `k1` (1.8-2.0) and lower `b` (0.3-0.5)
- **For long documents** (articles, books): Use lower `k1` (1.2-1.5) and higher `b` (0.7-0.9)
- **For balanced corpora**: Use default values (`k1=1.5`, `b=0.75`)

```python
# Example with custom parameters
bm25 = fastbm25.BM25(corpus, k1=2.0, b=0.8)
```

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
