#!/usr/bin/env python3
"""
Comprehensive performance testing suite for AI Cybersecurity Tool
Tests API performance under various load conditions and scenarios
"""

import time
import requests
import numpy as np
import joblib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PerformanceTest:
    """Comprehensive performance testing for AI Cybersecurity Tool API"""

    def __init__(self, api_url="http://localhost:5001"):
        self.api_url = api_url
        self.results = {
            "response_times": [],
            "throughput": [],
            "errors": 0,
            "error_details": [],
            "test_timestamps": [],
        }

        # Load feature names for realistic test data
        try:
            self.feature_names = joblib.load("models/feature_names.pkl")
        except FileNotFoundError:
            self.feature_names = [
                "Destination Port",
                "Flow Duration",
                "Total Fwd Packets",
                "Total Backward Packets",
                "Total Length of Fwd Packets",
            ]

    def create_realistic_features(self, attack_type="normal"):
        """Create realistic feature data for testing"""
        features = {}

        if attack_type == "ddos":
            # DDoS pattern: high packet count, short duration
            features["Destination Port"] = 80
            features["Flow Duration"] = 1000
            features["Total Fwd Packets"] = 10000
            features["Total Backward Packets"] = 0
            features["Total Length of Fwd Packets"] = 15000000
            features["Total Length of Bwd Packets"] = 0
            features["Fwd Packet Length Max"] = 1500
            features["Fwd Packet Length Min"] = 1500
            features["Fwd Packet Length Mean"] = 1500
            features["Fwd Packet Length Std"] = 0
            features["Bwd Packet Length Max"] = 0
            features["Bwd Packet Length Min"] = 0
            features["Bwd Packet Length Mean"] = 0
            features["Bwd Packet Length Std"] = 0
            features["Flow Bytes/s"] = 15000000
            features["Flow Packets/s"] = 10000
            features["Flow IAT Mean"] = 0.1
            features["Flow IAT Std"] = 0.01
            features["Flow IAT Max"] = 0.2
            features["Flow IAT Min"] = 0.05

        elif attack_type == "portscan":
            # Port scan pattern: single packet, short duration
            features["Destination Port"] = 22
            features["Flow Duration"] = 100
            features["Total Fwd Packets"] = 1
            features["Total Backward Packets"] = 0
            features["Total Length of Fwd Packets"] = 60
            features["Total Length of Bwd Packets"] = 0
            features["Fwd Packet Length Max"] = 60
            features["Fwd Packet Length Min"] = 60
            features["Fwd Packet Length Mean"] = 60
            features["Fwd Packet Length Std"] = 0
            features["Bwd Packet Length Max"] = 0
            features["Bwd Packet Length Min"] = 0
            features["Bwd Packet Length Mean"] = 0
            features["Bwd Packet Length Std"] = 0
            features["Flow Bytes/s"] = 600
            features["Flow Packets/s"] = 10
            features["Flow IAT Mean"] = 100
            features["Flow IAT Std"] = 0
            features["Flow IAT Max"] = 100
            features["Flow IAT Min"] = 100

        else:  # normal
            # Normal traffic pattern
            features["Destination Port"] = 80
            features["Flow Duration"] = 10000
            features["Total Fwd Packets"] = 100
            features["Total Backward Packets"] = 100
            features["Total Length of Fwd Packets"] = 10000
            features["Total Length of Bwd Packets"] = 10000
            features["Fwd Packet Length Max"] = 1500
            features["Fwd Packet Length Min"] = 64
            features["Fwd Packet Length Mean"] = 100
            features["Fwd Packet Length Std"] = 50
            features["Bwd Packet Length Max"] = 1500
            features["Bwd Packet Length Min"] = 64
            features["Bwd Packet Length Mean"] = 100
            features["Bwd Packet Length Std"] = 50
            features["Flow Bytes/s"] = 2000
            features["Flow Packets/s"] = 20
            features["Flow IAT Mean"] = 50
            features["Flow IAT Std"] = 10
            features["Flow IAT Max"] = 100
            features["Flow IAT Min"] = 10

        # Fill remaining features with default values
        for feature_name in self.feature_names:
            if feature_name not in features:
                if "Active" in feature_name:
                    features[feature_name] = 1000
                elif "Idle" in feature_name:
                    features[feature_name] = 100
                elif "IAT" in feature_name:
                    features[feature_name] = 50
                else:
                    features[feature_name] = 100

        return features

    def single_request(self, attack_type="normal"):
        """Make a single prediction request"""
        data = {
            "features": self.create_realistic_features(attack_type),
            "source_ip": f"192.168.1.{np.random.randint(1, 255)}",
            "attack_type": f"{attack_type}_test",
        }

        start_time = time.time()
        try:
            response = requests.post(f"{self.api_url}/predict", json=data, timeout=10)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result_data = response.json()
                return {
                    "success": True,
                    "time": elapsed,
                    "threat_score": result_data.get("threat_score", 0),
                    "threat_level": result_data.get("threat_level", "None"),
                }
            else:
                return {
                    "success": False,
                    "time": elapsed,
                    "status_code": response.status_code,
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            return {"success": False, "time": time.time() - start_time, "error": str(e)}

    def load_test(self, num_requests=100, max_workers=10, attack_type="normal"):
        """Perform load testing with realistic data"""
        print(
            f"🚀 Starting load test: {num_requests} requests with {max_workers} workers"
        )
        print(f"   Attack type: {attack_type}")

        start_time = time.time()
        successful_requests = 0
        threat_scores = []
        threat_levels = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.single_request, attack_type)
                for _ in range(num_requests)
            ]

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result["success"]:
                    self.results["response_times"].append(
                        result["time"] * 1000
                    )  # Convert to ms
                    self.results["test_timestamps"].append(time.time())
                    successful_requests += 1

                    if "threat_score" in result:
                        threat_scores.append(result["threat_score"])
                    if "threat_level" in result:
                        threat_levels.append(result["threat_level"])
                else:
                    self.results["errors"] += 1
                    self.results["error_details"].append(
                        result.get("error", "Unknown error")
                    )

                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"   Completed: {i + 1}/{num_requests} requests")

        total_time = time.time() - start_time

        # Calculate comprehensive metrics
        metrics = {
            "test_type": "load_test",
            "attack_type": attack_type,
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": self.results["errors"],
            "success_rate": successful_requests / num_requests * 100,
            "total_time": total_time,
            "throughput": successful_requests / total_time,
            "avg_response_time": (
                np.mean(self.results["response_times"])
                if self.results["response_times"]
                else 0
            ),
            "p50_response_time": (
                np.percentile(self.results["response_times"], 50)
                if self.results["response_times"]
                else 0
            ),
            "p95_response_time": (
                np.percentile(self.results["response_times"], 95)
                if self.results["response_times"]
                else 0
            ),
            "p99_response_time": (
                np.percentile(self.results["response_times"], 99)
                if self.results["response_times"]
                else 0
            ),
            "min_response_time": (
                np.min(self.results["response_times"])
                if self.results["response_times"]
                else 0
            ),
            "max_response_time": (
                np.max(self.results["response_times"])
                if self.results["response_times"]
                else 0
            ),
            "std_response_time": (
                np.std(self.results["response_times"])
                if self.results["response_times"]
                else 0
            ),
            "avg_threat_score": np.mean(threat_scores) if threat_scores else 0,
            "threat_level_distribution": (
                pd.Series(threat_levels).value_counts().to_dict()
                if threat_levels
                else {}
            ),
        }

        return metrics

    def stress_test(self, duration_seconds=60, attack_type="normal"):
        """Perform stress testing over time"""
        print(f"🔥 Starting stress test for {duration_seconds} seconds")
        print(f"   Attack type: {attack_type}")

        start_time = time.time()
        request_count = 0
        threat_scores = []

        while time.time() - start_time < duration_seconds:
            result = self.single_request(attack_type)

            if result["success"]:
                self.results["response_times"].append(result["time"] * 1000)
                if "threat_score" in result:
                    threat_scores.append(result["threat_score"])
            else:
                self.results["errors"] += 1
                self.results["error_details"].append(
                    result.get("error", "Unknown error")
                )

            request_count += 1

            # Progress indicator
            if request_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"   Completed: {request_count} requests in {elapsed:.1f}s")

        total_time = time.time() - start_time

        return {
            "test_type": "stress_test",
            "attack_type": attack_type,
            "total_requests": request_count,
            "successful_requests": request_count - self.results["errors"],
            "failed_requests": self.results["errors"],
            "success_rate": (request_count - self.results["errors"])
            / request_count
            * 100,
            "total_time": total_time,
            "throughput": (request_count - self.results["errors"]) / total_time,
            "avg_response_time": (
                np.mean(self.results["response_times"])
                if self.results["response_times"]
                else 0
            ),
            "avg_threat_score": np.mean(threat_scores) if threat_scores else 0,
        }

    def spike_test(self, base_requests=50, spike_requests=200, max_workers=20):
        """Test system behavior under sudden load spikes"""
        print(
            f"⚡ Starting spike test: {base_requests} base + {spike_requests} spike requests"
        )

        # Baseline load
        print("   Phase 1: Baseline load...")
        baseline_results = self.load_test(base_requests, max_workers // 2, "normal")

        # Clear results for spike test
        self.results["response_times"] = []
        self.results["errors"] = 0
        self.results["error_details"] = []

        # Spike load
        print("   Phase 2: Spike load...")
        spike_results = self.load_test(spike_requests, max_workers, "ddos")

        return {
            "test_type": "spike_test",
            "baseline": baseline_results,
            "spike": spike_results,
            "performance_degradation": {
                "response_time_increase": (
                    spike_results["avg_response_time"]
                    - baseline_results["avg_response_time"]
                )
                / baseline_results["avg_response_time"]
                * 100,
                "throughput_decrease": (
                    baseline_results["throughput"] - spike_results["throughput"]
                )
                / baseline_results["throughput"]
                * 100,
            },
        }

    def plot_results(self, save_path="performance_results.png"):
        """Plot comprehensive performance test results"""
        if not self.results["response_times"]:
            print("❌ No results to plot")
            return

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "AI Cybersecurity Tool - Performance Test Results",
            fontsize=16,
            fontweight="bold",
        )

        # Response time distribution
        axes[0, 0].hist(
            self.results["response_times"],
            bins=50,
            edgecolor="black",
            alpha=0.7,
            color="skyblue",
        )
        axes[0, 0].set_title("Response Time Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Response Time (ms)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(
            np.mean(self.results["response_times"]),
            color="red",
            linestyle="--",
            label=f'Mean: {np.mean(self.results["response_times"]):.1f}ms',
        )
        axes[0, 0].legend()

        # Response time over time
        axes[0, 1].plot(self.results["response_times"], alpha=0.7, color="green")
        axes[0, 1].set_title("Response Time Over Time", fontweight="bold")
        axes[0, 1].set_xlabel("Request Number")
        axes[0, 1].set_ylabel("Response Time (ms)")
        axes[0, 1].axhline(
            np.mean(self.results["response_times"]),
            color="red",
            linestyle="--",
            alpha=0.7,
        )

        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        percentile_values = [
            np.percentile(self.results["response_times"], p) for p in percentiles
        ]
        bars = axes[0, 2].bar(
            [f"P{p}" for p in percentiles], percentile_values, color="orange", alpha=0.7
        )
        axes[0, 2].set_title("Response Time Percentiles", fontweight="bold")
        axes[0, 2].set_ylabel("Response Time (ms)")

        # Add value labels on bars
        for bar, value in zip(bars, percentile_values):
            axes[0, 2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{value:.1f}ms",
                ha="center",
                va="bottom",
            )

        # Success rate pie chart
        success_rate = len(self.results["response_times"]) / (
            len(self.results["response_times"]) + self.results["errors"]
        )
        axes[1, 0].pie(
            [success_rate, 1 - success_rate],
            labels=["Success", "Failed"],
            autopct="%1.1f%%",
            colors=["green", "red"],
            startangle=90,
        )
        axes[1, 0].set_title("Request Success Rate", fontweight="bold")

        # Response time box plot
        axes[1, 1].boxplot(
            self.results["response_times"],
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
        )
        axes[1, 1].set_title("Response Time Distribution (Box Plot)", fontweight="bold")
        axes[1, 1].set_ylabel("Response Time (ms)")

        # Performance metrics summary
        axes[1, 2].axis("off")
        metrics_text = f"""
Performance Summary:
• Total Requests: {len(self.results['response_times']) + self.results['errors']}
• Successful: {len(self.results['response_times'])}
• Failed: {self.results['errors']}
• Success Rate: {success_rate:.1%}
• Avg Response Time: {np.mean(self.results['response_times']):.1f}ms
• P95 Response Time: {np.percentile(self.results['response_times'], 95):.1f}ms
• P99 Response Time: {np.percentile(self.results['response_times'], 99):.1f}ms
• Min Response Time: {np.min(self.results['response_times']):.1f}ms
• Max Response Time: {np.max(self.results['response_times']):.1f}ms
        """
        axes[1, 2].text(
            0.1,
            0.9,
            metrics_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Performance results saved to: {save_path}")
        plt.show()

    def save_results(self, results, filename="performance_results.json"):
        """Save test results to JSON file"""
        results["timestamp"] = datetime.now().isoformat()
        results["test_environment"] = {
            "api_url": self.api_url,
            "feature_count": len(self.feature_names),
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to: {filename}")

    def check_performance_requirements(self, results):
        """Check if performance meets requirements"""
        requirements = {
            "p95_response_time": 5000,  # 5 seconds in ms
            "success_rate": 95,  # 95% success rate
            "avg_response_time": 2000,  # 2 seconds average
        }

        print("\n📋 Performance Requirements Check:")
        print("=" * 40)

        passed = 0
        total = len(requirements)

        for metric, threshold in requirements.items():
            if metric in results:
                value = results[metric]
                status = "✅ PASS" if value <= threshold else "❌ FAIL"
                print(f"{metric}: {value:.2f} (threshold: {threshold}) {status}")
                if value <= threshold:
                    passed += 1
            else:
                print(f"{metric}: Not available")

        print(f"\nOverall: {passed}/{total} requirements met")

        if passed == total:
            print("🎉 All performance requirements met!")
            return True
        else:
            print("⚠️  Some performance requirements not met")
            return False


def main():
    """Main performance testing function"""
    print("🚀 AI Cybersecurity Tool - Performance Testing Suite")
    print("=" * 60)

    # Initialize tester
    tester = PerformanceTest()

    # Check API availability
    try:
        response = requests.get(f"{tester.api_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server not available. Please start the API first.")
            return
    except requests.exceptions.RequestException:
        print("❌ API server not available. Please start the API first.")
        return

    print("✅ API server is available")

    # Run comprehensive performance tests
    all_results = {}

    # 1. Load Test - Normal Traffic
    print("\n" + "=" * 60)
    print("TEST 1: LOAD TEST - NORMAL TRAFFIC")
    print("=" * 60)
    normal_load_results = tester.load_test(
        num_requests=100, max_workers=10, attack_type="normal"
    )
    all_results["normal_load"] = normal_load_results

    # Clear results for next test
    tester.results = {
        "response_times": [],
        "throughput": [],
        "errors": 0,
        "error_details": [],
        "test_timestamps": [],
    }

    # 2. Load Test - DDoS Traffic
    print("\n" + "=" * 60)
    print("TEST 2: LOAD TEST - DDOS TRAFFIC")
    print("=" * 60)
    ddos_load_results = tester.load_test(
        num_requests=100, max_workers=10, attack_type="ddos"
    )
    all_results["ddos_load"] = ddos_load_results

    # Clear results for next test
    tester.results = {
        "response_times": [],
        "throughput": [],
        "errors": 0,
        "error_details": [],
        "test_timestamps": [],
    }

    # 3. Stress Test
    print("\n" + "=" * 60)
    print("TEST 3: STRESS TEST - 30 SECONDS")
    print("=" * 60)
    stress_results = tester.stress_test(duration_seconds=30, attack_type="normal")
    all_results["stress_test"] = stress_results

    # Clear results for next test
    tester.results = {
        "response_times": [],
        "throughput": [],
        "errors": 0,
        "error_details": [],
        "test_timestamps": [],
    }

    # 4. Spike Test
    print("\n" + "=" * 60)
    print("TEST 4: SPIKE TEST")
    print("=" * 60)
    spike_results = tester.spike_test(
        base_requests=30, spike_requests=100, max_workers=15
    )
    all_results["spike_test"] = spike_results

    # Print comprehensive results
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PERFORMANCE RESULTS")
    print("=" * 60)

    for test_name, results in all_results.items():
        print(f"\n📊 {test_name.upper().replace('_', ' ')}:")
        if isinstance(results, dict) and "test_type" in results:
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

    # Check performance requirements
    print("\n" + "=" * 60)
    print("PERFORMANCE REQUIREMENTS CHECK")
    print("=" * 60)

    # Use the most comprehensive test results for requirements check
    main_results = normal_load_results
    tester.check_performance_requirements(main_results)

    # Plot results
    print("\n📊 Generating performance plots...")
    tester.plot_results("comprehensive_performance_results.png")

    # Save results
    tester.save_results(all_results, "comprehensive_performance_results.json")

    print("\n🎉 Performance testing completed!")


if __name__ == "__main__":
    main()
