#!/usr/bin/env python3
"""
Comprehensive test runner for fastbm25.

This script runs all tests and provides a summary of results.
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Add the test directory to the path
test_dir = Path(__file__).parent / "test"
sys.path.insert(0, str(test_dir))

def run_test(test_name, test_function):
    """Run a single test and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        test_function()
        end_time = time.time()
        print(f"\n✅ {test_name} PASSED ({end_time - start_time:.3f}s)")
        return True, end_time - start_time
    except Exception as e:
        end_time = time.time()
        print(f"\n❌ {test_name} FAILED ({end_time - start_time:.3f}s)")
        print(f"Error: {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
        return False, end_time - start_time

def test_basic_functionality():
    """Test basic BM25 functionality."""
    from test_module import test_bm25
    test_bm25()

def test_document_order():
    """Test that document order is preserved."""
    from order_test import test_document_order
    test_document_order()

def test_consistency():
    """Test that results are consistent across runs."""
    from controlled_test import test_parallel_vs_serial, test_performance_with_fixed_corpus
    
    print("Testing parallel vs serial consistency...")
    consistent = test_parallel_vs_serial()
    if not consistent:
        raise AssertionError("Parallel vs serial consistency test failed")
    
    print("\nTesting performance with fixed corpus...")
    test_performance_with_fixed_corpus()

def test_performance():
    """Test performance with various corpus sizes."""
    from performance_test import test_consistency, test_performance
    
    print("Testing BM25 consistency...")
    consistent = test_consistency()
    if not consistent:
        raise AssertionError("Consistency test failed")
    
    print("\nTesting performance...")
    test_performance()

def main():
    """Run all tests."""
    print("🚀 Starting fastbm25 Test Suite")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Define all tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Document Order Preservation", test_document_order),
        ("Consistency Tests", test_consistency),
        ("Performance Tests", test_performance),
    ]
    
    # Run tests
    results = []
    total_start_time = time.time()
    
    for test_name, test_function in tests:
        success, duration = run_test(test_name, test_function)
        results.append((test_name, success, duration))
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test_name, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name:<30} ({duration:.3f}s)")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_duration:.3f}s")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 {failed} test(s) failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
