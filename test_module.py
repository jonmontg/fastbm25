#!/usr/bin/env python3
"""
Test script for the fastbm25 module.
"""

import fastbm25

def test_bm25():
    """Test the BM25 implementation."""
    print("Testing fastbm25 module...")

    # Create a simple corpus
    corpus = [
        ["hello", "world"],
        ["hello", "python"],
        ["python", "programming", "language"],
        ["rust", "programming", "language"],
        ["hello", "rust", "world"]
    ]

    print(f"Corpus: {corpus}")

    # Initialize BM25
    bm25 = fastbm25.BM25(corpus)
    print("BM25 initialized successfully!")

    # Test query
    query = ["hello", "python"]
    print(f"Query: {query}")

    # Get scores
    scores = bm25.get_scores(query)
    print(f"Scores: {scores}")

    # Get top-k indices
    top_indices = bm25.get_top_k_indices(query, k=3)
    print(f"Top 3 indices: {top_indices}")

    # Test with different parameters
    bm25_custom = fastbm25.BM25(corpus, k1=2.0, b=0.8)
    scores_custom = bm25_custom.get_scores(query)
    print(f"Scores with custom params (k1=2.0, b=0.8): {scores_custom}")

    print("All tests passed!")

if __name__ == "__main__":
    test_bm25()
