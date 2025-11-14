#!/usr/bin/env python3
"""
Performance tests for new models
Compares prediction latency, batch performance, and concurrent handling
"""

import time
import requests
import numpy as np
import joblib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import statistics

API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"
MODELS_DIR = Path(__file__).parent / "models"


def load_feature_names():
    """Load new model feature names"""
    features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
    if features_path.exists():
        features = joblib.load(features_path)
        if not isinstance(features, list):
            features = list(features)
        return features
    return None


def create_test_features():
    """Create test features"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {
            "Destination Port": 80,
            "Flow Duration": 100000,
            "Total Fwd Packets": 10,
            "Total Backward Packets": 10,
        }
    
    features = {}
    for feat in feature_names[:30]:
        if "Port" in feat:
            features[feat] = 443
        elif "Duration" in feat:
            features[feat] = 100000
        elif "Packets" in feat:
            features[feat] = 10
        elif "Bytes" in feat:
            features[feat] = 2000
        else:
            features[feat] = np.random.uniform(0, 100)
    
    return features


def test_single_prediction_latency():
    """Test single prediction latency"""
    print("=" * 60)
    print("SINGLE PREDICTION LATENCY TEST")
    print("=" * 60)
    
    features = create_test_features()
    latencies = []
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    num_tests = 20
    print(f"Running {num_tests} predictions...")
    
    for i in range(num_tests):
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"features": features},
                headers=headers,
                timeout=10
            )
            end_time = time.time()
            
            if response.status_code == 200:
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            else:
                print(f"  Request {i+1} failed: {response.status_code}")
        except Exception as e:
            print(f"  Request {i+1} error: {e}")
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"\nResults:")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  Median: {median_latency:.2f} ms")
        print(f"  Min: {min_latency:.2f} ms")
        print(f"  Max: {max_latency:.2f} ms")
        print(f"  95th percentile: {p95_latency:.2f} ms")
        
        # Check if meets performance target (< 100ms)
        if avg_latency < 100:
            print(f"\n✅ PASS: Average latency ({avg_latency:.2f} ms) < 100 ms")
        else:
            print(f"\n⚠️  WARNING: Average latency ({avg_latency:.2f} ms) >= 100 ms")
        
        return avg_latency
    else:
        print("\n❌ No successful predictions")
        return None


def test_batch_prediction_performance():
    """Test batch prediction performance"""
    print("\n" + "=" * 60)
    print("BATCH PREDICTION PERFORMANCE TEST")
    print("=" * 60)
    
    features = create_test_features()
    batch_sizes = [10, 50, 100]
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        start_time = time.time()
        
        successful = 0
        for i in range(batch_size):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json={"features": features},
                    headers=headers,
                    timeout=10
                )
                if response.status_code == 200:
                    successful += 1
            except:
                pass
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = successful / total_time if total_time > 0 else 0
        
        print(f"  Successful: {successful}/{batch_size}")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Throughput: {throughput:.2f} predictions/s")
        print(f"  Avg per prediction: {(total_time / batch_size * 1000):.2f} ms")


def test_concurrent_request_handling():
    """Test concurrent request handling"""
    print("\n" + "=" * 60)
    print("CONCURRENT REQUEST HANDLING TEST")
    print("=" * 60)
    
    features = create_test_features()
    num_concurrent = 20
    
    def make_request():
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        }
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"features": features},
                headers=headers,
                timeout=10
            )
            end_time = time.time()
            return {
                "success": response.status_code == 200,
                "latency": (end_time - start_time) * 1000
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    print(f"Testing {num_concurrent} concurrent requests...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(make_request) for _ in range(num_concurrent)]
        results = [f.result() for f in futures]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    successful = sum(1 for r in results if r.get("success", False))
    latencies = [r["latency"] for r in results if r.get("success", False)]
    
    print(f"\nResults:")
    print(f"  Successful: {successful}/{num_concurrent}")
    print(f"  Total time: {total_time:.2f} s")
    if latencies:
        print(f"  Average latency: {statistics.mean(latencies):.2f} ms")
        print(f"  Max latency: {max(latencies):.2f} ms")
        print(f"  Min latency: {min(latencies):.2f} ms")
    
    if successful == num_concurrent:
        print(f"\n✅ PASS: All concurrent requests succeeded")
    else:
        print(f"\n⚠️  WARNING: {num_concurrent - successful} requests failed")


def test_memory_usage():
    """Test memory usage (basic check)"""
    print("\n" + "=" * 60)
    print("MEMORY USAGE TEST")
    print("=" * 60)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make several predictions
        features = create_test_features()
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        }
        
        for i in range(50):
            requests.post(
                f"{API_BASE_URL}/predict",
                json={"features": features},
                headers=headers,
                timeout=10
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        
        if memory_increase < 100:
            print(f"\n✅ PASS: Memory increase ({memory_increase:.2f} MB) < 100 MB")
        else:
            print(f"\n⚠️  WARNING: Memory increase ({memory_increase:.2f} MB) >= 100 MB")
    except ImportError:
        print("⚠️  psutil not available, skipping memory test")


def main():
    """Run all performance tests"""
    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not available")
            return
    except Exception as e:
        print(f"❌ API is not available: {e}")
        return
    
    # Run tests
    test_single_prediction_latency()
    test_batch_prediction_performance()
    test_concurrent_request_handling()
    test_memory_usage()
    
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

