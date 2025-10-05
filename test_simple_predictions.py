#!/usr/bin/env python3
"""
Simple prediction test with correct CICIDS2017 feature format (78 features)
Generates predictions to populate model performance metrics
"""
import requests
import json
import time
import random
import numpy as np

API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"

# All 78 CICIDS2017 feature names
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
    """Create realistic benign network traffic"""
    features = {}
    features['Destination Port'] = 443  # HTTPS
    features['Flow Duration'] = np.random.randint(100000, 500000)
    features['Total Fwd Packets'] = np.random.randint(5, 20)
    features['Total Backward Packets'] = np.random.randint(3, 15)
    features['Total Length of Fwd Packets'] = np.random.randint(500, 2000)
    features['Total Length of Bwd Packets'] = np.random.randint(400, 1800)
    features['Flow Bytes/s'] = np.random.uniform(1000, 5000)
    features['Flow Packets/s'] = np.random.uniform(10, 100)

    # Fill remaining features
    for feat in CICIDS_FEATURES:
        if feat not in features:
            features[feat] = np.random.uniform(0, 100)

    return features

def create_medium_threat():
    """Create medium threat pattern"""
    features = {}
    features['Destination Port'] = 80
    features['Flow Duration'] = np.random.randint(50000, 200000)
    features['Total Fwd Packets'] = np.random.randint(20, 80)
    features['Total Backward Packets'] = np.random.randint(5, 20)
    features['Flow Bytes/s'] = np.random.uniform(10000, 50000)
    features['Flow Packets/s'] = np.random.uniform(100, 500)

    # Fill remaining features
    for feat in CICIDS_FEATURES:
        if feat not in features:
            features[feat] = np.random.uniform(0, 150)

    return features

def create_high_threat():
    """Create high threat pattern (DDoS-like)"""
    features = {}
    features['Destination Port'] = 80
    features['Flow Duration'] = np.random.randint(10000, 50000)
    features['Total Fwd Packets'] = np.random.randint(100, 300)
    features['Total Backward Packets'] = np.random.randint(0, 5)
    features['Flow Bytes/s'] = np.random.uniform(100000, 500000)
    features['Flow Packets/s'] = np.random.uniform(1000, 5000)

    # Fill remaining features
    for feat in CICIDS_FEATURES:
        if feat not in features:
            features[feat] = np.random.uniform(0, 200)

    return features

def make_prediction(features, name="Test"):
    """Make a prediction with the correct 78-feature format"""

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
            print(f"   Model Predictions: {json.dumps(data.get('model_predictions', {}), indent=6)}")
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
    print("=" * 60)
    print("🧪 SIMPLE PREDICTION TEST (CICIDS2017)")
    print("=" * 60)
    print(f"API: {API_BASE_URL}")
    print(f"API Key: {API_KEY}")
    print(f"Features: {len(CICIDS_FEATURES)} CICIDS2017 features")
    print()

    # Test patterns with varying threat levels
    test_cases = [
        # Low threat (benign traffic)
        (create_benign_traffic, "Low Threat Pattern #1 (Benign)"),
        (create_benign_traffic, "Low Threat Pattern #2 (Benign)"),

        # Medium threat (moderate values)
        (create_medium_threat, "Medium Threat Pattern #1"),
        (create_medium_threat, "Medium Threat Pattern #2"),

        # High threat (DDoS-like)
        (create_high_threat, "High Threat Pattern #1 (DDoS-like)"),
        (create_high_threat, "High Threat Pattern #2 (DDoS-like)"),

        # Random patterns
        (create_benign_traffic, "Random Pattern #1"),
        (create_medium_threat, "Random Pattern #2"),
        (create_high_threat, "Random Pattern #3"),
        (create_benign_traffic, "Random Pattern #4"),
    ]

    success_count = 0

    for feature_generator, name in test_cases:
        features = feature_generator()
        result = make_prediction(features, name)
        if result:
            success_count += 1
        time.sleep(0.5)  # Brief delay between requests

    print("=" * 60)
    print(f"📊 Results: {success_count}/{len(test_cases)} successful predictions")
    print("=" * 60)
    print()

    # Check model performance
    print("📈 Fetching Model Performance Metrics...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/models/performance",
            headers={"X-API-Key": API_KEY},
            timeout=5
        )

        if response.status_code == 200:
            perf_data = response.json()
            print(f"\nTotal Predictions: {perf_data.get('total_predictions', 0)}")
            print(f"Healthy Models: {perf_data.get('healthy_models', 0)}/{perf_data.get('total_models', 0)}")
            print("\nPer-Model Metrics:")

            for model_name, metrics in perf_data.get('models', {}).items():
                print(f"\n  {model_name.upper()}:")
                print(f"    Status: {metrics.get('status', 'unknown')}")
                print(f"    Predictions: {metrics.get('predictions', 0)}")
                print(f"    Avg Confidence: {metrics.get('avg_confidence', 0):.4f}")
                print(f"    Avg Time: {metrics.get('avg_time_ms', 0):.2f}ms")
                print(f"    Contribution: {metrics.get('contribution_weight', 0):.1f}%")
        else:
            print(f"Failed to get performance metrics: {response.status_code}")
    except Exception as e:
        print(f"Error fetching performance: {e}")

    print("\n✅ Test complete!")
    print("🌐 Check your dashboard at http://localhost:3000")
    print("   - Model Performance should now show 'HEALTHY' status")
    print("   - Charts should display actual metrics")
    print()

if __name__ == "__main__":
    main()
