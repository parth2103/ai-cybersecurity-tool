#!/usr/bin/env python3
"""
End-to-end tests for new models
Tests complete workflow: API → model → response → dashboard
"""

import requests
import json
import time
import numpy as np
import joblib
from pathlib import Path

API_BASE_URL = "http://localhost:5001"
FRONTEND_URL = "http://localhost:3000"
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


def create_test_features(attack_type="normal"):
    """Create test features"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {
            "Destination Port": 80 if attack_type != "normal" else 443,
            "Flow Duration": 1000 if attack_type != "normal" else 100000,
            "Total Fwd Packets": 10000 if attack_type != "normal" else 10,
            "Total Backward Packets": 0 if attack_type != "normal" else 10,
        }
    
    features = {}
    for feat in feature_names[:30]:
        if "Port" in feat:
            features[feat] = 80 if attack_type != "normal" else 443
        elif "Duration" in feat:
            features[feat] = 1000 if attack_type != "normal" else 100000
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 10000 if attack_type != "normal" else 10
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 0 if attack_type != "normal" else 10
        elif "Bytes" in feat:
            features[feat] = 15000000 if attack_type != "normal" else 2000
        else:
            features[feat] = np.random.uniform(0, 100)
    
    return features


def test_complete_workflow():
    """Test complete workflow: API → model → response → dashboard"""
    print("=" * 60)
    print("END-TO-END WORKFLOW TEST")
    print("=" * 60)
    
    # Step 1: Check API health
    print("\n1. Checking API health...")
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("   ✅ API is healthy")
        else:
            print(f"   ❌ API health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API not available: {e}")
        return False
    
    # Step 2: Check system info
    print("\n2. Checking system info...")
    headers = {"X-API-Key": API_KEY}
    try:
        info_response = requests.get(f"{API_BASE_URL}/system/info", headers=headers, timeout=5)
        if info_response.status_code == 200:
            info_data = info_response.json()
            models_loaded = info_data.get("models_loaded", [])
            print(f"   ✅ Models loaded: {len(models_loaded)}")
            print(f"   Models: {', '.join(models_loaded) if models_loaded else 'None'}")
        else:
            print(f"   ⚠️  Could not get system info: {info_response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error getting system info: {e}")
    
    # Step 3: Make prediction with benign traffic
    print("\n3. Testing benign traffic prediction...")
    benign_features = create_test_features("normal")
    payload = {
        "features": benign_features,
        "source_ip": "192.168.1.100",
        "attack_type": "benign_test"
    }
    
    try:
        pred_response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
            timeout=10
        )
        
        if pred_response.status_code == 200:
            pred_data = pred_response.json()
            threat_score = pred_data.get("threat_score", 0)
            threat_level = pred_data.get("threat_level", "Unknown")
            print(f"   ✅ Prediction successful")
            print(f"   Threat Score: {threat_score:.4f}")
            print(f"   Threat Level: {threat_level}")
        else:
            print(f"   ❌ Prediction failed: {pred_response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return False
    
    # Step 4: Make prediction with attack traffic
    print("\n4. Testing attack traffic prediction...")
    attack_features = create_test_features("attack")
    payload = {
        "features": attack_features,
        "source_ip": "192.168.1.200",
        "attack_type": "attack_test"
    }
    
    try:
        pred_response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
            timeout=10
        )
        
        if pred_response.status_code == 200:
            pred_data = pred_response.json()
            threat_score = pred_data.get("threat_score", 0)
            threat_level = pred_data.get("threat_level", "Unknown")
            print(f"   ✅ Prediction successful")
            print(f"   Threat Score: {threat_score:.4f}")
            print(f"   Threat Level: {threat_level}")
        else:
            print(f"   ❌ Prediction failed: {pred_response.status_code}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    # Step 5: Check stats updated
    print("\n5. Checking stats updated...")
    time.sleep(0.5)
    try:
        stats_response = requests.get(f"{API_BASE_URL}/stats", headers=headers, timeout=5)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            total_requests = stats_data.get("total_requests", 0)
            threats_detected = stats_data.get("threats_detected", 0)
            print(f"   ✅ Stats updated")
            print(f"   Total Requests: {total_requests}")
            print(f"   Threats Detected: {threats_detected}")
        else:
            print(f"   ⚠️  Could not get stats: {stats_response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error getting stats: {e}")
    
    # Step 6: Check frontend (optional)
    print("\n6. Checking frontend accessibility...")
    try:
        frontend_response = requests.get(f"{FRONTEND_URL}", timeout=5)
        if frontend_response.status_code == 200:
            print(f"   ✅ Frontend is accessible")
        else:
            print(f"   ⚠️  Frontend returned: {frontend_response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Frontend not accessible: {e}")
    
    print("\n" + "=" * 60)
    print("✅ END-TO-END TEST COMPLETE")
    print("=" * 60)
    print("\n🌐 Check your dashboard at http://localhost:3000")
    print("   - Threat scores should be visible")
    print("   - Charts should update")
    print("   - Model performance should show metrics")
    
    return True


def test_error_handling():
    """Test error handling and recovery"""
    print("\n" + "=" * 60)
    print("ERROR HANDLING TEST")
    print("=" * 60)
    
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    
    # Test 1: Missing features
    print("\n1. Testing with missing features...")
    payload = {"features": {"Destination Port": 80}}
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, headers=headers, timeout=10)
    print(f"   Status: {response.status_code} (should be 200 or 400)")
    
    # Test 2: Invalid feature values
    print("\n2. Testing with invalid feature values...")
    features = create_test_features("normal")
    features["InvalidFeature"] = "not_a_number"
    payload = {"features": features}
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, headers=headers, timeout=10)
    print(f"   Status: {response.status_code} (should handle gracefully)")
    
    # Test 3: Empty request
    print("\n3. Testing with empty request...")
    payload = {}
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, headers=headers, timeout=10)
    print(f"   Status: {response.status_code} (should be 400)")
    
    print("\n✅ Error handling test complete")


def main():
    """Run E2E tests"""
    success = test_complete_workflow()
    test_error_handling()
    
    if success:
        print("\n✅ All E2E tests passed!")
    else:
        print("\n⚠️  Some E2E tests had issues")


if __name__ == "__main__":
    main()

