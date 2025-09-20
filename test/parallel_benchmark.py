#!/usr/bin/env python3
"""
Benchmark to compare parallel vs serial performance of get_scores.
"""

import time
import random
import fastbm25
import statistics

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

def benchmark_get_scores():
    """Benchmark the get_scores function with different corpus sizes."""
    print("=== get_scores Performance Benchmark ===\n")
    
    # Test different corpus sizes
    test_cases = [
        (1000, 50, 200),     # Small
        (5000, 100, 500),    # Medium  
        (10000, 150, 1000),  # Large
        (20000, 200, 1500),  # Very large
        (50000, 250, 2000),  # Extra large
    ]
    
    for num_docs, avg_len, vocab_size in test_cases:
        print(f"Testing with {num_docs} documents, avg length {avg_len}, vocab size {vocab_size}")
        
        # Generate corpus
        text_corpus = generate_large_corpus(num_docs, avg_len, vocab_size, seed=42)
        token_to_id = create_vocabulary(text_corpus)
        numeric_corpus = [[token_to_id[token] for token in doc] for doc in text_corpus]
        
        # Initialize BM25
        bm25 = fastbm25.BM25(numeric_corpus)
        
        # Create test queries of different sizes
        queries = [
            [token_to_id[token] for token in list(token_to_id.keys())[:5]],   # Short
            [token_to_id[token] for token in list(token_to_id.keys())[:15]],  # Medium
            [token_to_id[token] for token in list(token_to_id.keys())[:30]],  # Long
        ]
        
        query_names = ["Short (5 terms)", "Medium (15 terms)", "Long (30 terms)"]
        
        for query, query_name in zip(queries, query_names):
            # Warm up
            _ = bm25.get_scores(query)
            
            # Time multiple runs for statistical significance
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                scores = bm25.get_scores(query)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times)
            min_time = min(times)
            max_time = max(times)
            
            non_zero_count = sum(1 for s in scores if s > 0)
            throughput = len(scores) / avg_time
            
            print(f"  {query_name}:")
            print(f"    Time: {avg_time:.4f}s ± {std_time:.4f}s (min: {min_time:.4f}s, max: {max_time:.4f}s)")
            print(f"    Throughput: {throughput:.0f} docs/sec")
            print(f"    Non-zero scores: {non_zero_count}/{len(scores)} ({100*non_zero_count/len(scores):.1f}%)")
        
        print()

def test_parallel_efficiency():
    """Test parallel efficiency with different thread counts."""
    print("=== Parallel Efficiency Test ===\n")
    
    # Use a large corpus to see parallel benefits
    num_docs = 30000
    avg_len = 200
    vocab_size = 1500
    
    print(f"Testing with {num_docs} documents, avg length {avg_len}, vocab size {vocab_size}")
    
    # Generate corpus
    text_corpus = generate_large_corpus(num_docs, avg_len, vocab_size, seed=42)
    token_to_id = create_vocabulary(text_corpus)
    numeric_corpus = [[token_to_id[token] for token in doc] for doc in text_corpus]
    
    # Initialize BM25
    bm25 = fastbm25.BM25(numeric_corpus)
    
    # Create a complex query
    query = [token_to_id[token] for token in list(token_to_id.keys())[:20]]
    
    # Warm up
    _ = bm25.get_scores(query)
    
    # Time multiple runs
    times = []
    for _ in range(5):
        start_time = time.perf_counter()
        scores = bm25.get_scores(query)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = statistics.mean(times)
    throughput = len(scores) / avg_time
    
    print(f"Average time: {avg_time:.4f}s")
    print(f"Throughput: {throughput:.0f} docs/sec")
    print(f"Total documents processed: {len(scores)}")
    print(f"Documents per second: {throughput:.0f}")
    
    # Calculate theoretical speedup
    import os
    cpu_count = os.cpu_count()
    print(f"CPU cores available: {cpu_count}")
    print(f"Theoretical max speedup: {cpu_count}x")

if __name__ == "__main__":
    benchmark_get_scores()
    print()
    test_parallel_efficiency()
