#!/usr/bin/env python3
"""
Performance test for multithreaded BM25 implementation.
"""

import time
import random
import fastbm25

def generate_large_corpus(num_docs=10000, avg_doc_len=100, vocab_size=1000, seed=42):
    """Generate a large corpus for performance testing."""
    print(f"Generating corpus with {num_docs} documents, avg length {avg_doc_len}, vocab size {vocab_size}")
    
    # Create vocabulary
    vocab = [f"word_{i}" for i in range(vocab_size)]
    
    # Set seed for reproducible results
    random.seed(seed)
    
    # Generate documents
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

def test_consistency():
    """Test that the same corpus produces identical results."""
    print("=== Testing BM25 Consistency ===\n")
    
    # Generate a small test corpus
    text_corpus = generate_large_corpus(100, 20, 50, seed=123)
    token_to_id = create_vocabulary(text_corpus)
    numeric_corpus = [[token_to_id[token] for token in doc] for doc in text_corpus]
    
    # Test multiple initializations with the same corpus
    results = []
    for i in range(3):
        bm25 = fastbm25.BM25(numeric_corpus)
        query = [token_to_id[token] for token in list(token_to_id.keys())[:3]]
        scores = bm25.get_scores(query)
        top_indices = bm25.get_top_k_indices(query, k=5)
        results.append((scores, top_indices))
        print(f"Run {i+1}: top_indices = {top_indices}")
    
    # Check consistency
    first_scores, first_indices = results[0]
    consistent = all(scores == first_scores and indices == first_indices 
                    for scores, indices in results[1:])
    
    if consistent:
        print("✅ SUCCESS: All runs produce identical results!")
    else:
        print("❌ FAILURE: Results differ between runs!")
    
    return consistent

def test_performance():
    """Test performance with different corpus sizes."""
    print("=== BM25 Performance Test ===\n")
    
    # Test different corpus sizes
    test_cases = [
        (1000, 50, 200),    # Small: 1K docs, avg 50 words, 200 vocab
        (5000, 100, 500),   # Medium: 5K docs, avg 100 words, 500 vocab  
        (10000, 150, 1000), # Large: 10K docs, avg 150 words, 1K vocab
    ]
    
    for test_idx, (num_docs, avg_len, vocab_size) in enumerate(test_cases):
        print(f"Testing with {num_docs} documents, avg length {avg_len}, vocab size {vocab_size}")
        
        # Generate corpus with unique seed for each test case
        start_time = time.time()
        text_corpus = generate_large_corpus(num_docs, avg_len, vocab_size, seed=42 + test_idx)
        corpus_gen_time = time.time() - start_time
        
        # Create vocabulary
        start_time = time.time()
        token_to_id = create_vocabulary(text_corpus)
        vocab_time = time.time() - start_time
        
        # Convert to numeric IDs
        start_time = time.time()
        numeric_corpus = [[token_to_id[token] for token in doc] for doc in text_corpus]
        conversion_time = time.time() - start_time
        
        # Initialize BM25 (this is where multithreading helps)
        start_time = time.time()
        bm25 = fastbm25.BM25(numeric_corpus)
        bm25_init_time = time.time() - start_time
        
        # Test query performance with deterministic query
        query_tokens = list(token_to_id.keys())[:5]  # Use first 5 tokens for consistency
        query = [token_to_id[token] for token in query_tokens]
        
        start_time = time.time()
        scores = bm25.get_scores(query)
        query_time = time.time() - start_time
        
        start_time = time.time()
        top_indices = bm25.get_top_k_indices(query, k=10)
        top_k_time = time.time() - start_time
        
        # Test consistency with multiple initializations
        print(f"  Testing consistency...")
        consistent_results = []
        for _ in range(3):
            bm25_test = fastbm25.BM25(numeric_corpus)
            scores_test = bm25_test.get_scores(query)
            top_test = bm25_test.get_top_k_indices(query, k=10)
            consistent_results.append((scores_test, top_test))
        
        # Check if all results are identical
        first_scores, first_top = consistent_results[0]
        is_consistent = all(s == first_scores and t == first_top 
                           for s, t in consistent_results[1:])
        print(f"  Consistency check: {'✅ PASS' if is_consistent else '❌ FAIL'}")
        
        print(f"  Corpus generation: {corpus_gen_time:.3f}s")
        print(f"  Vocabulary creation: {vocab_time:.3f}s")
        print(f"  Text to numeric conversion: {conversion_time:.3f}s")
        print(f"  BM25 initialization: {bm25_init_time:.3f}s")
        print(f"  Query scoring: {query_time:.3f}s")
        print(f"  Top-k retrieval: {top_k_time:.3f}s")
        print(f"  Total time: {corpus_gen_time + vocab_time + conversion_time + bm25_init_time:.3f}s")
        print(f"  Non-zero scores: {sum(1 for s in scores if s > 0)}/{len(scores)}")
        print(f"  Top indices: {top_indices[:5]}...")
        print()

if __name__ == "__main__":
    # First test consistency
    consistent = test_consistency()
    print()
    
    if consistent:
        # Only run performance test if consistency test passes
        test_performance()
    else:
        print("Skipping performance test due to consistency issues.")
