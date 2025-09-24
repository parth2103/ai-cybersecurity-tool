#!/usr/bin/env python3
"""
Quick performance test for AI Cybersecurity Tool
Lightweight performance validation for CI/CD pipelines
"""

import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class QuickPerformanceTest:
    """Quick performance test for CI/CD validation"""
    
    def __init__(self, api_url="http://localhost:5001"):
        self.api_url = api_url
    
    def create_simple_features(self):
        """Create simple feature data for testing"""
        return {
            'Destination Port': 80,
            'Flow Duration': 1000,
            'Total Fwd Packets': 100,
            'Total Backward Packets': 100,
            'Total Length of Fwd Packets': 10000,
            'Total Length of Bwd Packets': 10000,
            'Fwd Packet Length Max': 1500,
            'Fwd Packet Length Min': 64,
            'Fwd Packet Length Mean': 100,
            'Fwd Packet Length Std': 50,
            'Bwd Packet Length Max': 1500,
            'Bwd Packet Length Min': 64,
            'Bwd Packet Length Mean': 100,
            'Bwd Packet Length Std': 50,
            'Flow Bytes/s': 2000,
            'Flow Packets/s': 20,
            'Flow IAT Mean': 50,
            'Flow IAT Std': 10,
            'Flow IAT Max': 100,
            'Flow IAT Min': 10
        }
    
    def single_request(self):
        """Make a single prediction request"""
        data = {
            'features': self.create_simple_features(),
            'source_ip': f'192.168.1.{np.random.randint(1, 255)}',
            'attack_type': 'performance_test'
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=data,
                timeout=10
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                return {'success': True, 'time': elapsed * 1000}  # Convert to ms
            else:
                return {'success': False, 'time': elapsed * 1000, 'status': response.status_code}
        except Exception as e:
            return {'success': False, 'time': (time.time() - start_time) * 1000, 'error': str(e)}
    
    def quick_load_test(self, num_requests=20, max_workers=5):
        """Quick load test for CI/CD validation"""
        print(f"🚀 Quick Performance Test: {num_requests} requests with {max_workers} workers")
        
        start_time = time.time()
        response_times = []
        errors = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.single_request) for _ in range(num_requests)]
            
            for future in futures:
                result = future.result()
                if result['success']:
                    response_times.append(result['time'])
                else:
                    errors += 1
        
        total_time = time.time() - start_time
        successful_requests = num_requests - errors
        
        # Calculate metrics
        metrics = {
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'failed_requests': errors,
            'success_rate': successful_requests / num_requests * 100,
            'total_time': total_time,
            'throughput': successful_requests / total_time,
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
            'max_response_time': np.max(response_times) if response_times else 0
        }
        
        return metrics
    
    def check_requirements(self, metrics):
        """Check if performance meets basic requirements"""
        requirements = {
            'success_rate': 90,  # 90% success rate
            'p95_response_time': 5000,  # 5 seconds
            'avg_response_time': 3000   # 3 seconds average
        }
        
        print("\n📋 Performance Requirements Check:")
        print("-" * 40)
        
        passed = 0
        total = len(requirements)
        
        for metric, threshold in requirements.items():
            if metric in metrics:
                value = metrics[metric]
                if metric == 'success_rate':
                    status = "✅ PASS" if value >= threshold else "❌ FAIL"
                    print(f"{metric}: {value:.1f}% (min: {threshold}%) {status}")
                else:
                    status = "✅ PASS" if value <= threshold else "❌ FAIL"
                    print(f"{metric}: {value:.1f}ms (max: {threshold}ms) {status}")
                
                if (metric == 'success_rate' and value >= threshold) or (metric != 'success_rate' and value <= threshold):
                    passed += 1
            else:
                print(f"{metric}: Not available")
        
        print(f"\nOverall: {passed}/{total} requirements met")
        return passed == total

def main():
    """Main quick performance test function"""
    print("⚡ AI Cybersecurity Tool - Quick Performance Test")
    print("=" * 50)
    
    # Initialize tester
    tester = QuickPerformanceTest()
    
    # Check API availability
    try:
        response = requests.get(f"{tester.api_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server not available. Please start the API first.")
            return False
    except requests.exceptions.RequestException:
        print("❌ API server not available. Please start the API first.")
        return False
    
    print("✅ API server is available")
    
    # Run quick performance test
    print("\n🧪 Running quick performance test...")
    metrics = tester.quick_load_test(num_requests=20, max_workers=5)
    
    # Print results
    print("\n📊 Performance Results:")
    print("-" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key or 'throughput' in key:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.1f}ms")
        else:
            print(f"{key}: {value}")
    
    # Check requirements
    passed = tester.check_requirements(metrics)
    
    if passed:
        print("\n🎉 Quick performance test PASSED!")
        return True
    else:
        print("\n⚠️  Quick performance test FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
