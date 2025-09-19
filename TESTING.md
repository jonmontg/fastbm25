# Testing fastbm25

This document describes how to run tests for the fastbm25 library.

## Quick Start

Run all tests:
```bash
# Using Poetry
poetry run test

# Using Make
make test

# Direct execution
poetry run python test_runner.py
```

## Test Categories

### 1. Basic Functionality Tests
Tests the core BM25 algorithm functionality:
- Document scoring
- Top-k retrieval
- Parameter handling
- Edge cases

```bash
make test-basic
# or
poetry run python test/test_module.py
```

### 2. Document Order Preservation Tests
Verifies that multithreaded processing preserves document order:
- Multiple runs with same corpus
- Consistent results across runs
- Document index mapping

```bash
make test-order
# or
poetry run python test/order_test.py
```

### 3. Performance Tests
Tests performance and consistency with large corpora:
- Various corpus sizes (1K, 5K, 10K documents)
- Performance benchmarking
- Consistency verification
- Multithreading benefits

```bash
make test-performance
# or
poetry run python test/performance_test.py
```

### 4. Consistency Tests
Verifies that parallel and serial implementations produce identical results:
- Identical corpus comparison
- Score consistency
- Index consistency

```bash
poetry run python test/controlled_test.py
```

## Test Structure

```
test/
├── test_module.py          # Basic functionality tests
├── order_test.py           # Document order preservation
├── performance_test.py     # Performance and consistency
└── controlled_test.py      # Parallel vs serial consistency
```

## Running Individual Tests

You can run any individual test file:

```bash
# Basic functionality
poetry run python test/test_module.py

# Document order preservation
poetry run python test/order_test.py

# Performance tests
poetry run python test/performance_test.py

# Consistency tests
poetry run python test/controlled_test.py
```

## Test Output

The test runner provides detailed output including:
- ✅/❌ Status for each test
- Execution time for each test
- Detailed error messages for failures
- Summary statistics

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines:

```bash
# Install dependencies
poetry install

# Build the project
poetry run maturin develop

# Run all tests
poetry run test
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the project is built with `poetry run maturin develop`
2. **Rust compilation errors**: Ensure Rust is installed and up to date
3. **Performance test failures**: These may be due to system load or resource constraints

### Debug Mode

For more verbose output, you can run individual test files directly:

```bash
poetry run python -v test/test_module.py
```

## Adding New Tests

To add new tests:

1. Create a new test file in the `test/` directory
2. Follow the naming convention: `test_*.py`
3. Add the test to `test_runner.py` if it should be part of the main test suite
4. Update this documentation

## Test Requirements

- Python 3.7+
- Rust toolchain
- Poetry for dependency management
- Sufficient memory for performance tests (recommended: 4GB+)
