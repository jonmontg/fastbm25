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
        ["hello", "rust", "world"],
    ]

    tokens = list(set([token for doc in corpus for token in doc]))
    token_to_code = {token: code for code, token in enumerate(tokens)}
    encoded_corpus = [[token_to_code[token] for token in doc] for doc in corpus]

    tokenize = lambda x: token_to_code.get(x, len(tokens))

    print(f"Corpus: {corpus}")
    print(f"Encoded corpus: {encoded_corpus}")

    # Initialize BM25
    bm25 = fastbm25.BM25(encoded_corpus)
    print("BM25 initialized successfully!")

    # Test query
    query = ["hello", "python"]
    print(f"Query: {query}")

    # Get scores
    scores = bm25.get_scores([tokenize(word) for word in query])
    assert scores == [
        0.06257709693956749,
        0.43803967857697246,
        0.3146862644658826,
        0.0,
        0.052447710744313765,
    ]

    # Get top-k indices
    top_indices = bm25.get_top_k_indices([tokenize(word) for word in query], k=3)
    assert top_indices == [1, 2, 0]

    # Test with different parameters
    bm25_custom = fastbm25.BM25(encoded_corpus, k1=2.0, b=0.8)
    scores_custom = bm25_custom.get_scores([token_to_code[word] for word in query])
    assert scores_custom == [
        0.06394940169701416,
        0.44764581187909913,
        0.31095775422339583,
        0.0,
        0.05182629237056598,
    ]

    # Query with terms not in corpus
    query = ["absent", "terms"]
    indices = bm25.get_top_k_indices([tokenize(word) for word in query], k=3)
    scores = bm25.get_scores([tokenize(word) for word in query])
    assert indices == []
    assert scores == [0.0, 0.0, 0.0, 0.0, 0.0]

    print("All tests passed!")


if __name__ == "__main__":
    test_bm25()
