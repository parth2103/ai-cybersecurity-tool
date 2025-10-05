#!/usr/bin/env python3
"""
Test predictions using actual CICIDS2017 features (78 features)
Verifies models work after proper retraining
"""
import requests
import json
import time
import numpy as np

API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"

# All 78 CICIDS2017 feature names (from feature_names.pkl)
CICIDS_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
    'Idle Std', 'Idle Max', 'Idle Min'
]

def create_benign_traffic():
    """Create realistic benign network traffic features"""
    features = {}
    features['Destination Port'] = 443  # HTTPS
    features['Flow Duration'] = np.random.randint(100000, 500000)
    features['Total Fwd Packets'] = np.random.randint(5, 20)
    features['Total Backward Packets'] = np.random.randint(3, 15)
    features['Total Length of Fwd Packets'] = np.random.randint(500, 2000)
    features['Total Length of Bwd Packets'] = np.random.randint(400, 1800)
    features['Fwd Packet Length Max'] = np.random.randint(100, 300)
    features['Fwd Packet Length Min'] = np.random.randint(20, 60)
    features['Fwd Packet Length Mean'] = np.random.uniform(50, 150)
    features['Fwd Packet Length Std'] = np.random.uniform(10, 50)
    features['Bwd Packet Length Max'] = np.random.randint(100, 300)
    features['Bwd Packet Length Min'] = np.random.randint(20, 60)
    features['Bwd Packet Length Mean'] = np.random.uniform(50, 150)
    features['Bwd Packet Length Std'] = np.random.uniform(10, 50)
    features['Flow Bytes/s'] = np.random.uniform(1000, 5000)
    features['Flow Packets/s'] = np.random.uniform(10, 100)
    features['Flow IAT Mean'] = np.random.uniform(1000, 10000)
    features['Flow IAT Std'] = np.random.uniform(500, 5000)
    features['Flow IAT Max'] = np.random.uniform(5000, 20000)
    features['Flow IAT Min'] = np.random.uniform(100, 1000)

    # Fill remaining features with realistic random values
    for feat in CICIDS_FEATURES:
        if feat not in features:
            features[feat] = np.random.uniform(0, 100)

    return features

def create_malicious_traffic():
    """Create realistic malicious network traffic features (DDoS-like)"""
    features = {}
    features['Destination Port'] = 80  # HTTP
    features['Flow Duration'] = np.random.randint(10000, 50000)  # Shorter duration
    features['Total Fwd Packets'] = np.random.randint(50, 200)  # Many packets
    features['Total Backward Packets'] = np.random.randint(0, 5)  # Few responses
    features['Total Length of Fwd Packets'] = np.random.randint(5000, 20000)
    features['Total Length of Bwd Packets'] = np.random.randint(0, 500)
    features['Fwd Packet Length Max'] = np.random.randint(50, 100)
    features['Fwd Packet Length Min'] = np.random.randint(40, 60)
    features['Fwd Packet Length Mean'] = np.random.uniform(40, 80)
    features['Fwd Packet Length Std'] = np.random.uniform(5, 20)
    features['Bwd Packet Length Max'] = np.random.randint(40, 80)
    features['Bwd Packet Length Min'] = np.random.randint(20, 40)
    features['Bwd Packet Length Mean'] = np.random.uniform(30, 60)
    features['Bwd Packet Length Std'] = np.random.uniform(5, 15)
    features['Flow Bytes/s'] = np.random.uniform(50000, 200000)  # High throughput
    features['Flow Packets/s'] = np.random.uniform(500, 2000)  # Many packets/sec
    features['Flow IAT Mean'] = np.random.uniform(100, 1000)  # Short intervals
    features['Flow IAT Std'] = np.random.uniform(50, 500)
    features['Flow IAT Max'] = np.random.uniform(1000, 5000)
    features['Flow IAT Min'] = np.random.uniform(10, 100)

    # Fill remaining features
    for feat in CICIDS_FEATURES:
        if feat not in features:
            features[feat] = np.random.uniform(0, 100)

    return features

def make_prediction(features, name="Test"):
    """Make a prediction with 78 CICIDS2017 features"""

    payload = {"features": features}

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ {name}")
            print(f"   Threat Score: {data.get('threat_score', 0):.4f} ({data.get('threat_score', 0)*100:.1f}%)")
            print(f"   Threat Level: {data.get('threat_level', 'Unknown')}")
            print(f"   Model Predictions:")
            for model, score in data.get('model_predictions', {}).items():
                print(f"      {model}: {score:.4f}")
            print()
            return data
        else:
            print(f"❌ {name} failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ {name} error: {e}")
        return None

def main():
    print("=" * 70)
    print("🧪 CICIDS2017 FEATURE PREDICTION TEST")
    print("=" * 70)
    print(f"API: {API_BASE_URL}")
    print(f"Features: {len(CICIDS_FEATURES)} (CICIDS2017)")
    print()

    # Test benign traffic
    print("📊 Testing Benign Traffic Patterns...")
    print("-" * 70)
    for i in range(3):
        features = create_benign_traffic()
        make_prediction(features, f"Benign Traffic #{i+1}")
        time.sleep(0.5)

    # Test malicious traffic
    print("\n⚠️  Testing Malicious Traffic Patterns (DDoS-like)...")
    print("-" * 70)
    for i in range(3):
        features = create_malicious_traffic()
        make_prediction(features, f"Malicious Traffic #{i+1}")
        time.sleep(0.5)

    # Check model performance
    print("\n" + "=" * 70)
    print("📈 Fetching Model Performance Metrics...")
    print("=" * 70)
    try:
        response = requests.get(
            f"{API_BASE_URL}/models/performance",
            headers={"X-API-Key": API_KEY},
            timeout=5
        )

        if response.status_code == 200:
            perf_data = response.json()
            print(f"\n✅ Total Predictions: {perf_data.get('total_predictions', 0)}")
            print(f"✅ Healthy Models: {perf_data.get('healthy_models', 0)}/{perf_data.get('total_models', 0)}")
            print("\n📊 Per-Model Performance:")

            for model_name, metrics in perf_data.get('models', {}).items():
                status_emoji = "✅" if metrics.get('status') == 'healthy' else "⚠️"
                print(f"\n  {status_emoji} {model_name.upper()}")
                print(f"      Status: {metrics.get('status', 'unknown')}")
                print(f"      Predictions: {metrics.get('predictions', 0)}")
                print(f"      Avg Confidence: {metrics.get('avg_confidence', 0):.4f}")
                print(f"      Avg Time: {metrics.get('avg_time_ms', 0):.2f}ms")
                print(f"      Contribution: {metrics.get('contribution_weight', 0):.1f}%")
                print(f"      Available: {'Yes' if metrics.get('available') else 'No'}")
        else:
            print(f"❌ Failed to get performance metrics: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error fetching performance: {e}")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)
    print("\n🎯 Next Steps:")
    print("   1. Check dashboard at http://localhost:3000")
    print("   2. Verify model status shows 'HEALTHY'")
    print("   3. Check Model Performance Monitor displays metrics")
    print("   4. Verify Attention Visualizer shows feature importance")
    print()

if __name__ == "__main__":
    main()
