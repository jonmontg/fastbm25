#!/usr/bin/env python3
"""
Test to verify that document order is preserved in multithreaded BM25.
"""

import fastbm25
import random

def test_document_order():
    """Test that document order is preserved."""
    print("Testing document order preservation...")
    
    # Create a simple corpus
    corpus = [
        [0, 1, 2],  # doc 0
        [1, 2, 3],  # doc 1
        [2, 3, 4],  # doc 2
        [3, 4, 5],  # doc 3
        [4, 5, 0],  # doc 4
    ]
    
    # Initialize BM25 multiple times
    results = []
    for i in range(5):
        bm25 = fastbm25.BM25(corpus)
        query = [1, 2]
        scores = bm25.get_scores(query)
        top_indices = bm25.get_top_k_indices(query, k=3)
        results.append((scores, top_indices))
        print(f"Run {i+1}: scores={[f'{s:.3f}' for s in scores]}, top_indices={top_indices}")
    
    # Check that all runs produce identical results
    first_scores, first_indices = results[0]
    all_identical = True
    
    for i, (scores, indices) in enumerate(results[1:], 1):
        if scores != first_scores or indices != first_indices:
            print(f"ERROR: Run {i+1} differs from run 1!")
            all_identical = False
    
    if all_identical:
        print("✅ SUCCESS: All runs produce identical results - document order is preserved!")
    else:
        print("❌ FAILURE: Results differ between runs - document order is not preserved!")
    
    return all_identical

if __name__ == "__main__":
    test_document_order()
