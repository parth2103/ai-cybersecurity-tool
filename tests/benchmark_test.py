#!/usr/bin/env python3
"""
Benchmark testing for AI Cybersecurity Tool
Compares performance across different scenarios and configurations
"""

import time
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BenchmarkTest:
    """Benchmark testing for performance comparison"""
    
    def __init__(self, api_url="http://localhost:5001"):
        self.api_url = api_url
        self.benchmark_results = {}
    
    def create_benchmark_features(self, complexity="simple"):
        """Create features of different complexity levels"""
        base_features = {
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
        
        if complexity == "complex":
            # Add more complex features
            base_features.update({
                'Active Mean': 1000,
                'Active Std': 100,
                'Active Max': 2000,
                'Active Min': 500,
                'Idle Mean': 100,
                'Idle Std': 50,
                'Idle Max': 200,
                'Idle Min': 10
            })
        
        return base_features
    
    def benchmark_request(self, complexity="simple"):
        """Make a benchmark request"""
        data = {
            'features': self.create_benchmark_features(complexity),
            'source_ip': f'192.168.1.{np.random.randint(1, 255)}',
            'attack_type': f'benchmark_{complexity}'
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
                result_data = response.json()
                return {
                    'success': True,
                    'time': elapsed * 1000,  # Convert to ms
                    'threat_score': result_data.get('threat_score', 0),
                    'complexity': complexity
                }
            else:
                return {
                    'success': False,
                    'time': elapsed * 1000,
                    'status_code': response.status_code,
                    'complexity': complexity
                }
        except Exception as e:
            return {
                'success': False,
                'time': (time.time() - start_time) * 1000,
                'error': str(e),
                'complexity': complexity
            }
    
    def run_benchmark(self, complexity="simple", num_requests=50, max_workers=5):
        """Run benchmark test for specific complexity"""
        print(f"🏃 Running benchmark: {complexity} complexity, {num_requests} requests")
        
        start_time = time.time()
        response_times = []
        threat_scores = []
        errors = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.benchmark_request, complexity) for _ in range(num_requests)]
            
            for future in futures:
                result = future.result()
                if result['success']:
                    response_times.append(result['time'])
                    if 'threat_score' in result:
                        threat_scores.append(result['threat_score'])
                else:
                    errors += 1
        
        total_time = time.time() - start_time
        successful_requests = num_requests - errors
        
        benchmark_result = {
            'complexity': complexity,
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'failed_requests': errors,
            'success_rate': successful_requests / num_requests * 100,
            'total_time': total_time,
            'throughput': successful_requests / total_time,
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'p50_response_time': np.percentile(response_times, 50) if response_times else 0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
            'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
            'min_response_time': np.min(response_times) if response_times else 0,
            'max_response_time': np.max(response_times) if response_times else 0,
            'std_response_time': np.std(response_times) if response_times else 0,
            'avg_threat_score': np.mean(threat_scores) if threat_scores else 0
        }
        
        self.benchmark_results[complexity] = benchmark_result
        return benchmark_result
    
    def compare_benchmarks(self):
        """Compare benchmark results across different complexities"""
        if len(self.benchmark_results) < 2:
            print("❌ Need at least 2 benchmark results to compare")
            return
        
        print("\n📊 Benchmark Comparison:")
        print("=" * 60)
        
        # Create comparison DataFrame
        comparison_data = []
        for complexity, results in self.benchmark_results.items():
            comparison_data.append({
                'Complexity': complexity,
                'Success Rate (%)': results['success_rate'],
                'Avg Response Time (ms)': results['avg_response_time'],
                'P95 Response Time (ms)': results['p95_response_time'],
                'Throughput (req/s)': results['throughput'],
                'Avg Threat Score': results['avg_threat_score']
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Performance analysis
        print("\n🔍 Performance Analysis:")
        print("-" * 30)
        
        if 'simple' in self.benchmark_results and 'complex' in self.benchmark_results:
            simple = self.benchmark_results['simple']
            complex_result = self.benchmark_results['complex']
            
            response_time_increase = (complex_result['avg_response_time'] - simple['avg_response_time']) / simple['avg_response_time'] * 100
            throughput_decrease = (simple['throughput'] - complex_result['throughput']) / simple['throughput'] * 100
            
            print(f"Response time increase (simple → complex): {response_time_increase:.1f}%")
            print(f"Throughput decrease (simple → complex): {throughput_decrease:.1f}%")
            
            if response_time_increase < 50:
                print("✅ Performance impact is acceptable (<50% increase)")
            else:
                print("⚠️  Performance impact is significant (>50% increase)")
    
    def plot_benchmark_comparison(self, save_path='benchmark_comparison.png'):
        """Plot benchmark comparison results"""
        if len(self.benchmark_results) < 2:
            print("❌ Need at least 2 benchmark results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AI Cybersecurity Tool - Benchmark Comparison', fontsize=16, fontweight='bold')
        
        complexities = list(self.benchmark_results.keys())
        
        # Response time comparison
        avg_times = [self.benchmark_results[c]['avg_response_time'] for c in complexities]
        p95_times = [self.benchmark_results[c]['p95_response_time'] for c in complexities]
        
        x = np.arange(len(complexities))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, avg_times, width, label='Average', alpha=0.8)
        axes[0, 0].bar(x + width/2, p95_times, width, label='P95', alpha=0.8)
        axes[0, 0].set_title('Response Time Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Complexity')
        axes[0, 0].set_ylabel('Response Time (ms)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(complexities)
        axes[0, 0].legend()
        
        # Throughput comparison
        throughputs = [self.benchmark_results[c]['throughput'] for c in complexities]
        axes[0, 1].bar(complexities, throughputs, alpha=0.8, color='green')
        axes[0, 1].set_title('Throughput Comparison', fontweight='bold')
        axes[0, 1].set_xlabel('Complexity')
        axes[0, 1].set_ylabel('Throughput (requests/second)')
        
        # Success rate comparison
        success_rates = [self.benchmark_results[c]['success_rate'] for c in complexities]
        axes[1, 0].bar(complexities, success_rates, alpha=0.8, color='blue')
        axes[1, 0].set_title('Success Rate Comparison', fontweight='bold')
        axes[1, 0].set_xlabel('Complexity')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_ylim(0, 100)
        
        # Threat score comparison
        threat_scores = [self.benchmark_results[c]['avg_threat_score'] for c in complexities]
        axes[1, 1].bar(complexities, threat_scores, alpha=0.8, color='orange')
        axes[1, 1].set_title('Average Threat Score Comparison', fontweight='bold')
        axes[1, 1].set_xlabel('Complexity')
        axes[1, 1].set_ylabel('Average Threat Score')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Benchmark comparison saved to: {save_path}")
        plt.show()

def main():
    """Main benchmark testing function"""
    print("🏃 AI Cybersecurity Tool - Benchmark Testing")
    print("=" * 50)
    
    # Initialize benchmark tester
    benchmark = BenchmarkTest()
    
    # Check API availability
    try:
        response = requests.get(f"{benchmark.api_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server not available. Please start the API first.")
            return
    except requests.exceptions.RequestException:
        print("❌ API server not available. Please start the API first.")
        return
    
    print("✅ API server is available")
    
    # Run benchmarks for different complexities
    print("\n🧪 Running benchmark tests...")
    
    # Simple complexity benchmark
    simple_results = benchmark.run_benchmark("simple", num_requests=50, max_workers=5)
    
    # Complex complexity benchmark
    complex_results = benchmark.run_benchmark("complex", num_requests=50, max_workers=5)
    
    # Compare results
    benchmark.compare_benchmarks()
    
    # Plot comparison
    print("\n📊 Generating benchmark comparison plots...")
    benchmark.plot_benchmark_comparison()
    
    print("\n🎉 Benchmark testing completed!")

if __name__ == "__main__":
    main()
