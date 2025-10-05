#!/usr/bin/env python3
"""
Test attention explainer with real CICIDS2017 features
"""
import requests
import json
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

def create_test_traffic():
    """Create test traffic with realistic features"""
    features = {}
    features['Destination Port'] = 80
    features['Flow Duration'] = 50000
    features['Total Fwd Packets'] = 100
    features['Total Backward Packets'] = 2
    features['Total Length of Fwd Packets'] = 10000
    features['Total Length of Bwd Packets'] = 200
    features['Flow Bytes/s'] = 100000
    features['Flow Packets/s'] = 1000
    features['Flow IAT Mean'] = 500

    # Fill remaining features
    for feat in CICIDS_FEATURES:
        if feat not in features:
            features[feat] = np.random.uniform(0, 100)

    return features

def test_explain_endpoint():
    """Test the /explain endpoint with CICIDS features"""

    print("=" * 70)
    print("🔍 ATTENTION EXPLAINER TEST")
    print("=" * 70)

    # Check health
    print("\n1️⃣ Checking explainer health...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/explain/health",
            headers={"X-API-Key": API_KEY},
            timeout=5
        )

        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Explainer Ready: {health.get('explainer_ready')}")
            print(f"   ✅ Baseline Model: {health.get('baseline_model_loaded')}")
            print(f"   ✅ Feature Names: {health.get('feature_names_loaded')}")
            print(f"   ✅ Features Count: {health.get('num_features')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Test explanation
    print("\n2️⃣ Testing explanation generation...")
    features = create_test_traffic()

    payload = {"features": features}
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/explain",
            json=payload,
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Explanation generated successfully!")
            print(f"\n   📊 Results:")
            print(f"      Threat Level: {data.get('threat_level')}")
            print(f"      Prediction Score: {data.get('prediction', 0):.4f}")
            print(f"\n   🎯 Top Features by Attention Weight:")

            top_features = data.get('top_features', [])
            for i, feat_tuple in enumerate(top_features[:10], 1):
                if isinstance(feat_tuple, (list, tuple)) and len(feat_tuple) >= 2:
                    feat_name = feat_tuple[0]
                    weight = feat_tuple[1]
                    print(f"      {i}. {feat_name}: {weight:.4f}")
                else:
                    print(f"      {i}. {feat_tuple}")

            print(f"\n   📝 Explanation:")
            explanation = data.get('explanation', '')
            for line in explanation.split('\n'):
                if line.strip():
                    print(f"      {line}")

            # Check visualization data
            viz_data = data.get('visualization_data', {})
            if viz_data:
                print(f"\n   📈 Visualization Data:")
                print(f"      Features: {len(viz_data.get('features', []))}")
                print(f"      Weights: {len(viz_data.get('weights', []))}")

        else:
            print(f"   ❌ Explanation failed: {response.status_code}")
            print(f"      Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ ATTENTION EXPLAINER TEST COMPLETE!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    test_explain_endpoint()
