"""
Performance Testing Script for Product Review Analyzer

This script tests the API latency and throughput of the deployed system.
Run after containers are up: python test_performance.py
"""

import requests
import time
import statistics

API_URL = "http://localhost:8000"

def test_product_list():
    """Test the /products endpoint"""
    print("Testing /products endpoint...")
    start = time.time()
    response = requests.get(f"{API_URL}/products")
    elapsed = (time.time() - start) * 1000

    if response.status_code == 200:
        products = response.json()
        print(f"SUCCESS: Retrieved {len(products)} products in {elapsed:.2f}ms")
        return products
    else:
        print(f"FAILED: Status {response.status_code}")
        return []

def test_analysis_latency(asin, num_runs=3):
    """Test analysis endpoint latency"""
    print(f"\nTesting /analyze/{asin} endpoint ({num_runs} runs)...")
    latencies = []

    for i in range(num_runs):
        start = time.time()
        response = requests.get(f"{API_URL}/analyze/{asin}")
        elapsed = (time.time() - start) * 1000

        if response.status_code == 200:
            latencies.append(elapsed)
            data = response.json()
            print(f"  Run {i+1}: {elapsed:.2f}ms (analyzed {data['analyzed_count']} reviews)")
        else:
            print(f"  Run {i+1}: FAILED - Status {response.status_code}")

    if latencies:
        print(f"\nLATENCY STATISTICS:")
        print(f"   Mean: {statistics.mean(latencies):.2f}ms")
        print(f"   Median: {statistics.median(latencies):.2f}ms")
        print(f"   Min: {min(latencies):.2f}ms")
        print(f"   Max: {max(latencies):.2f}ms")

        if len(latencies) > 1:
            print(f"   Std Dev: {statistics.stdev(latencies):.2f}ms")

    return latencies

def test_caching_effect(asin):
    """Test if caching improves response time"""
    print(f"\nTesting caching effect for {asin}...")

    # First call (cold cache)
    print("  Cold cache (1st call)...")
    start = time.time()
    requests.get(f"{API_URL}/analyze/{asin}")
    cold_latency = (time.time() - start) * 1000

    # Second call (warm cache)
    print("  Warm cache (2nd call)...")
    start = time.time()
    requests.get(f"{API_URL}/analyze/{asin}")
    warm_latency = (time.time() - start) * 1000

    speedup = cold_latency / warm_latency
    print(f"\nCACHE PERFORMANCE:")
    print(f"   Cold: {cold_latency:.2f}ms")
    print(f"   Warm: {warm_latency:.2f}ms")
    print(f"   Speedup: {speedup:.2f}x")

def main():
    print("="*60)
    print("PERFORMANCE TEST SUITE - Product Review Analyzer")
    print("="*60)

    # Test 1: Product list
    products = test_product_list()

    if not products:
        print("\nCannot continue without products. Check if backend is running.")
        return

    # Select a test product (first one)
    test_asin = products[0]['asin']
    print(f"\nUsing test product: {test_asin}")

    # Test 2: Analysis latency
    test_analysis_latency(test_asin, num_runs=3)

    # Test 3: Cache effect
    if len(products) > 1:
        test_caching_effect(products[1]['asin'])

    print("\n" + "="*60)
    print("PERFORMANCE TESTS COMPLETED")
    print("="*60)
    print("\nTo check backend logs:")
    print("  docker-compose logs backend | grep MONITORING")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to API at", API_URL)
        print("Make sure Docker containers are running:")
        print("  docker-compose up")
