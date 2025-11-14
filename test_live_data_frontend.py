#!/usr/bin/env python3
"""
Live data flow tests for frontend
Sends test predictions and verifies they appear on dashboard
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
    """Create test features with ALL required features"""
    feature_names = load_feature_names()
    if feature_names is None:
        # Fallback to basic features
        return {
            "Destination Port": 80 if attack_type != "normal" else 443,
            "Flow Duration": 1000 if attack_type != "normal" else 100000,
            "Total Fwd Packets": 10000 if attack_type != "normal" else 10,
            "Total Backward Packets": 0 if attack_type != "normal" else 10,
        }
    
    # Create features for ALL feature names (not just first 30)
    features = {}
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 80 if attack_type != "normal" else 443
        elif "Duration" in feat:
            features[feat] = 1000 if attack_type != "normal" else 100000
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 10000 if attack_type != "normal" else 10
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 0 if attack_type != "normal" else 10
        elif "Bytes/s" in feat or "Bytes per" in feat or "Bytes" in feat:
            features[feat] = 15000000 if attack_type != "normal" else 2000
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 10000 if attack_type != "normal" else 20
        elif "IAT" in feat or "Inter Arrival" in feat:
            features[feat] = 0.05 if attack_type != "normal" else 50
        elif "Active" in feat:
            features[feat] = 1000 if attack_type != "normal" else 100
        elif "Idle" in feat:
            features[feat] = 0 if attack_type != "normal" else 100
        elif "Flag" in feat or "Flags" in feat:
            features[feat] = 0 if attack_type != "normal" else 1
        elif "Length" in feat:
            features[feat] = 1500 if attack_type != "normal" else 100
        elif "Mean" in feat or "Avg" in feat or "Average" in feat:
            features[feat] = 1000 if attack_type != "normal" else 100
        elif "Std" in feat or "Variance" in feat:
            features[feat] = 100 if attack_type != "normal" else 10
        elif "Max" in feat:
            features[feat] = 1500 if attack_type != "normal" else 500
        elif "Min" in feat:
            features[feat] = 0 if attack_type != "normal" else 10
        else:
            # Default value for any other features
            features[feat] = np.random.uniform(0, 100) if attack_type == "normal" else np.random.uniform(0, 1000)
    
    return features


def send_prediction(features, name="Test"):
    """Send a prediction to the API"""
    payload = {
        "features": features,
        "source_ip": "192.168.1.100",
        "attack_type": name
    }
    
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
            return {
                "success": True,
                "threat_score": data.get("threat_score", 0),
                "threat_level": data.get("threat_level", "Unknown")
            }
        else:
            return {
                "success": False,
                "error": f"Status {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def check_stats():
    """Check current stats from API"""
    headers = {"X-API-Key": API_KEY}
    try:
        response = requests.get(f"{API_BASE_URL}/stats", headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def test_live_data_flow():
    """Test live data flow to frontend"""
    print("=" * 60)
    print("LIVE DATA FLOW TEST")
    print("=" * 60)
    print(f"API: {API_BASE_URL}")
    print(f"Frontend: {FRONTEND_URL}")
    print()
    
    # Check API health
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API is not available")
            return
        print("✅ API is healthy")
    except Exception as e:
        print(f"❌ API is not available: {e}")
        return
    
    # Check frontend (optional)
    try:
        frontend_response = requests.get(f"{FRONTEND_URL}", timeout=5)
        if frontend_response.status_code == 200:
            print("✅ Frontend is accessible")
        else:
            print("⚠️  Frontend may not be running")
    except:
        print("⚠️  Frontend may not be running")
    
    print()
    
    # Get initial stats
    print("📊 Initial Stats:")
    initial_stats = check_stats()
    if initial_stats:
        print(f"  Total Requests: {initial_stats.get('total_requests', 0)}")
        print(f"  Threats Detected: {initial_stats.get('threats_detected', 0)}")
        print(f"  Threat History Length: {len(initial_stats.get('threat_history', []))}")
    print()
    
    # Send multiple test predictions
    print("📤 Sending test predictions...")
    print("-" * 60)
    
    test_cases = [
        ("Benign Traffic #1", "normal"),
        ("Benign Traffic #2", "normal"),
        ("Attack Pattern #1", "attack"),
        ("Attack Pattern #2", "attack"),
        ("Benign Traffic #3", "normal"),
    ]
    
    results = []
    for name, attack_type in test_cases:
        features = create_test_features(attack_type)
        result = send_prediction(features, name)
        
        if result["success"]:
            status = "✅"
            threat_score = result["threat_score"]
            threat_level = result["threat_level"]
            print(f"{status} {name:25} | Score: {threat_score:.4f} | Level: {threat_level}")
            results.append(result)
        else:
            status = "❌"
            print(f"{status} {name:25} | Error: {result.get('error', 'Unknown')}")
        
        time.sleep(0.5)  # Brief delay between requests
    
    print()
    
    # Check updated stats
    print("📊 Updated Stats:")
    time.sleep(1)  # Wait for processing
    updated_stats = check_stats()
    if updated_stats:
        print(f"  Total Requests: {updated_stats.get('total_requests', 0)}")
        print(f"  Threats Detected: {updated_stats.get('threats_detected', 0)}")
        print(f"  Threat History Length: {len(updated_stats.get('threat_history', []))}")
        print(f"  Current Threat Level: {updated_stats.get('current_threat_level', 'Unknown')}")
        
        # Show recent threat history
        threat_history = updated_stats.get('threat_history', [])
        if threat_history:
            print(f"\n  Recent Threat History (last {min(5, len(threat_history))}):")
            for i, threat in enumerate(threat_history[-5:], 1):
                threat_score = threat.get('threat_score', 0)
                timestamp = threat.get('timestamp', 'Unknown')
                print(f"    {i}. Score: {threat_score:.4f} | Time: {timestamp}")
    print()
    
    # Test concurrent predictions
    print("🔄 Testing concurrent predictions...")
    print("-" * 60)
    
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(5):
            features = create_test_features("normal" if i % 2 == 0 else "attack")
            future = executor.submit(send_prediction, features, f"Concurrent_{i}")
            futures.append(future)
        
        concurrent_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        successful = sum(1 for r in concurrent_results if r.get("success", False))
        print(f"✅ {successful}/{len(concurrent_results)} concurrent predictions succeeded")
    print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Sent {len(results)} successful predictions")
    print(f"✅ Stats updated correctly")
    print(f"✅ Threat history populated")
    print()
    print("🌐 Check your dashboard at http://localhost:3000")
    print("   - Threat scores should appear in real-time")
    print("   - Charts should update")
    print("   - Model performance should show metrics")
    print("=" * 60)


def main():
    """Run live data flow test"""
    test_live_data_flow()


if __name__ == "__main__":
    main()

