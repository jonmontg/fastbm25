#!/usr/bin/env python3
"""
Performance test specifically for the parallelized get_scores function.
"""

import time
import random
import fastbm25


def generate_large_corpus(num_docs=10000, avg_doc_len=100, vocab_size=1000, seed=42):
    """Generate a large corpus for performance testing."""
    random.seed(seed)
    vocab = [f"word_{i}" for i in range(vocab_size)]

    corpus = []
    for _ in range(num_docs):
        doc_len = random.randint(avg_doc_len // 2, avg_doc_len * 2)
        doc = [random.choice(vocab) for _ in range(doc_len)]
        corpus.append(doc)

    return corpus


def create_vocabulary(corpus):
    """Create vocabulary mapping from text corpus."""
    tokens = list(set([token for doc in corpus for token in doc]))
    token_to_id = {token: token_id for token_id, token in enumerate(tokens)}
    return token_to_id


def test_score_performance():
    """Test the performance of the parallelized get_scores function."""
    print("=== get_scores Performance Test ===\n")

    # Test different corpus sizes
    test_cases = [
        (1000, 50, 200),  # Small
        (5000, 100, 500),  # Medium
        (10000, 150, 1000),  # Large
        (20000, 200, 1500),  # Very large
    ]

    for num_docs, avg_len, vocab_size in test_cases:
        print(
            f"Testing with {num_docs} documents, avg length {avg_len}, vocab size {vocab_size}"
        )

        # Generate corpus
        text_corpus = generate_large_corpus(num_docs, avg_len, vocab_size, seed=42)
        token_to_id = create_vocabulary(text_corpus)
        numeric_corpus = [[token_to_id[token] for token in doc] for doc in text_corpus]

        # Initialize BM25
        bm25 = fastbm25.BM25(numeric_corpus)

        # Create different types of queries
        queries = [
            [
                token_to_id[token] for token in list(token_to_id.keys())[:3]
            ],  # Short query
            [
                token_to_id[token] for token in list(token_to_id.keys())[:10]
            ],  # Medium query
            [
                token_to_id[token] for token in list(token_to_id.keys())[:20]
            ],  # Long query
        ]

        query_names = [
            "Short query (3 terms)",
            "Medium query (10 terms)",
            "Long query (20 terms)",
        ]

        for query, query_name in zip(queries, query_names):
            # Warm up
            _ = bm25.get_scores(query)

            # Time multiple runs
            times = []
            for _ in range(5):
                start_time = time.time()
                scores = bm25.get_scores(query)
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            non_zero_count = sum(1 for s in scores if s > 0)

            print(f"  {query_name}:")
            print(f"    Average time: {avg_time:.4f}s")
            print(f"    Min time: {min_time:.4f}s")
            print(f"    Max time: {max_time:.4f}s")
            print(f"    Non-zero scores: {non_zero_count}/{len(scores)}")
            print(f"    Throughput: {len(scores)/avg_time:.0f} docs/sec")

        print()


def test_score_consistency():
    """Test that parallelized get_scores produces consistent results."""
    print("=== get_scores Consistency Test ===\n")

    # Generate a test corpus
    text_corpus = generate_large_corpus(1000, 50, 200, seed=123)
    token_to_id = create_vocabulary(text_corpus)
    numeric_corpus = [[token_to_id[token] for token in doc] for doc in text_corpus]

    bm25 = fastbm25.BM25(numeric_corpus)
    query = [token_to_id[token] for token in list(token_to_id.keys())[:5]]

    # Run multiple times and check consistency
    results = []
    for i in range(10):
        scores = bm25.get_scores(query)
        results.append(scores)

    # Check if all results are identical
    first_scores = results[0]
    all_identical = all(scores == first_scores for scores in results[1:])

    if all_identical:
        print("✅ SUCCESS: All get_scores runs produce identical results!")
    else:
        print("❌ FAILURE: get_scores results differ between runs!")
        for i, scores in enumerate(results[1:], 1):
            if scores != first_scores:
                print(f"  Run {i+1} differs from Run 1")

    return all_identical


if __name__ == "__main__":
    test_score_consistency()
    print()
    test_score_performance()
