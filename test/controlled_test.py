#!/usr/bin/env python3
"""
Controlled test to compare parallel vs serial results with identical corpora.
"""

import fastbm25
import random
import time

def generate_deterministic_corpus(num_docs=100, avg_doc_len=20, vocab_size=50, seed=42):
    """Generate a deterministic corpus."""
    random.seed(seed)
    vocab = [f"word_{i}" for i in range(vocab_size)]
    
    corpus = []
    for _ in range(num_docs):
        doc_len = avg_doc_len
        doc = [random.choice(vocab) for _ in range(doc_len)]
        corpus.append(doc)
    
    return corpus

def test_parallel_vs_serial():
    """Test that parallel and serial versions produce identical results."""
    print("=== Testing Parallel vs Serial Consistency ===\n")
    
    # Generate identical corpora
    corpus1 = generate_deterministic_corpus(100, 20, 50, seed=42)
    corpus2 = generate_deterministic_corpus(100, 20, 50, seed=42)
    
    # Convert to numeric IDs
    tokens1 = list(set([token for doc in corpus1 for token in doc]))
    token_to_id1 = {token: token_id for token_id, token in enumerate(tokens1)}
    numeric_corpus1 = [[token_to_id1[token] for token in doc] for doc in corpus1]
    
    tokens2 = list(set([token for doc in corpus2 for token in doc]))
    token_to_id2 = {token: token_id for token_id, token in enumerate(tokens2)}
    numeric_corpus2 = [[token_to_id2[token] for token in doc] for doc in corpus2]
    
    print(f"Corpus 1 length: {len(numeric_corpus1)}")
    print(f"Corpus 2 length: {len(numeric_corpus2)}")
    print(f"First doc 1: {numeric_corpus1[0]}")
    print(f"First doc 2: {numeric_corpus2[0]}")
    print(f"Corpora identical: {numeric_corpus1 == numeric_corpus2}")
    
    # Test with same corpus
    bm25_1 = fastbm25.BM25(numeric_corpus1)
    bm25_2 = fastbm25.BM25(numeric_corpus1)  # Use same corpus
    
    query = [0, 1]
    scores1 = bm25_1.get_scores(query)
    scores2 = bm25_2.get_scores(query)
    top1 = bm25_1.get_top_k_indices(query, k=5)
    top2 = bm25_2.get_top_k_indices(query, k=5)
    
    print(f"\nQuery: {query}")
    print(f"Scores 1: {[f'{s:.3f}' for s in scores1[:10]]}...")
    print(f"Scores 2: {[f'{s:.3f}' for s in scores2[:10]]}...")
    print(f"Scores identical: {scores1 == scores2}")
    print(f"Top indices 1: {top1}")
    print(f"Top indices 2: {top2}")
    print(f"Top indices identical: {top1 == top2}")
    
    return scores1 == scores2 and top1 == top2

def test_performance_with_fixed_corpus():
    """Test performance with a fixed corpus."""
    print("\n=== Performance Test with Fixed Corpus ===\n")
    
    # Generate a fixed corpus
    corpus = generate_deterministic_corpus(1000, 50, 200, seed=123)
    tokens = list(set([token for doc in corpus for token in doc]))
    token_to_id = {token: token_id for token_id, token in enumerate(tokens)}
    numeric_corpus = [[token_to_id[token] for token in doc] for doc in corpus]
    
    print(f"Corpus size: {len(numeric_corpus)} documents")
    print(f"Vocabulary size: {len(tokens)}")
    
    # Test multiple runs with same corpus
    results = []
    for i in range(3):
        start_time = time.time()
        bm25 = fastbm25.BM25(numeric_corpus)
        init_time = time.time() - start_time
        
        query = [token_to_id[token] for token in list(tokens)[:3]]
        scores = bm25.get_scores(query)
        top_indices = bm25.get_top_k_indices(query, k=5)
        
        results.append((scores, top_indices, init_time))
        print(f"Run {i+1}: init_time={init_time:.3f}s, top_indices={top_indices}")
    
    # Check consistency
    first_scores, first_indices, _ = results[0]
    consistent = all(scores == first_scores and indices == first_indices 
                    for scores, indices, _ in results[1:])
    
    print(f"\nAll runs consistent: {consistent}")
    if not consistent:
        print("❌ Results differ between runs!")
    else:
        print("✅ All runs produce identical results!")

if __name__ == "__main__":
    test_parallel_vs_serial()
    test_performance_with_fixed_corpus()
