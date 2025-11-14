#!/usr/bin/env python3
"""
Test IoT attack patterns for new models
Tests DoS, DDoS, Mirai, Brute Force, Recon, Spoofing, and IIoT attacks
"""

import requests
import json
import numpy as np
import joblib
from pathlib import Path

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


def create_iot_dos_attack():
    """Create DoS attack pattern (IoT-IDAD 2024)"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {}
    
    features = {}
    
    # DoS pattern: high packet rate, short duration, overwhelming target
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 80  # HTTP
        elif "Duration" in feat:
            features[feat] = 1000  # Short duration
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 50000  # High packet count
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 0  # No response
        elif "Bytes/s" in feat or "Bytes per" in feat:
            features[feat] = 50000000  # High throughput
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 50000  # High packet rate
        elif "IAT" in feat or "Inter Arrival" in feat:
            features[feat] = 0.02  # Very short intervals
        else:
            features[feat] = np.random.uniform(0, 1000)
    
    return features


def create_iot_ddos_attack():
    """Create DDoS attack pattern (IoT-IDAD 2024)"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {}
    
    features = {}
    
    # DDoS pattern: massive packet flood from multiple sources
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 80
        elif "Duration" in feat:
            features[feat] = 500  # Very short
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 100000  # Massive packet count
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 0
        elif "Bytes/s" in feat or "Bytes per" in feat:
            features[feat] = 150000000  # Very high throughput
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 100000  # Very high packet rate
        elif "IAT" in feat or "Inter Arrival" in feat:
            features[feat] = 0.005  # Extremely short intervals
        else:
            features[feat] = np.random.uniform(0, 2000)
    
    return features


def create_mirai_botnet_attack():
    """Create Mirai botnet attack pattern (IoT-IDAD 2024)"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {}
    
    features = {}
    
    # Mirai pattern: coordinated attack from IoT devices, steady traffic
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 23  # Telnet (Mirai targets)
        elif "Duration" in feat:
            features[feat] = 50000  # Medium duration
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 10000  # Steady packet stream
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 100  # Some response
        elif "Bytes/s" in feat or "Bytes per" in feat:
            features[feat] = 10000000  # Moderate throughput
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 200  # Steady rate
        elif "IAT" in feat or "Inter Arrival" in feat:
            features[feat] = 5  # Regular intervals
        else:
            features[feat] = np.random.uniform(0, 500)
    
    return features


def create_brute_force_attack():
    """Create Brute Force attack pattern (IoT-IDAD 2024)"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {}
    
    features = {}
    
    # Brute Force pattern: repeated connection attempts
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 22  # SSH
        elif "Duration" in feat:
            features[feat] = 1000  # Short attempts
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 50  # Multiple attempts
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 50  # Responses
        elif "Bytes/s" in feat or "Bytes per" in feat:
            features[feat] = 5000  # Low throughput
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 100  # Moderate rate
        elif "IAT" in feat or "Inter Arrival" in feat:
            features[feat] = 10  # Regular intervals
        else:
            features[feat] = np.random.uniform(0, 100)
    
    return features


def create_recon_attack():
    """Create Recon attack pattern (IoT-IDAD 2024)"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {}
    
    features = {}
    
    # Recon pattern: scanning, probing, single packets to multiple ports
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 22  # SSH (common target)
        elif "Duration" in feat:
            features[feat] = 100  # Very short
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 1  # Single packet
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 0  # No response (port closed)
        elif "Bytes/s" in feat or "Bytes per" in feat:
            features[feat] = 60  # Very low
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 10  # Low rate
        elif "IAT" in feat or "Inter Arrival" in feat:
            features[feat] = 100  # Longer intervals
        else:
            features[feat] = np.random.uniform(0, 50)
    
    return features


def create_spoofing_attack():
    """Create Spoofing attack pattern (IoT-IDAD 2024)"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {}
    
    features = {}
    
    # Spoofing pattern: fake source, unusual patterns
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 53  # DNS
        elif "Duration" in feat:
            features[feat] = 5000  # Medium
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 1000  # Moderate
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 0  # No legitimate response
        elif "Bytes/s" in feat or "Bytes per" in feat:
            features[feat] = 50000  # Moderate
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 200  # Moderate rate
        elif "IAT" in feat or "Inter Arrival" in feat:
            features[feat] = 5  # Regular but suspicious
        else:
            features[feat] = np.random.uniform(0, 200)
    
    return features


def create_iiot_attack():
    """Create IIoT attack pattern (CICAPT-IIOT specific)"""
    feature_names = load_feature_names()
    if feature_names is None:
        return {}
    
    features = {}
    
    # IIoT pattern: Industrial control system attack
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 502  # Modbus (common IIoT protocol)
        elif "Duration" in feat:
            features[feat] = 20000  # Longer duration
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 5000  # Command packets
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 5000  # Responses
        elif "Bytes/s" in feat or "Bytes per" in feat:
            features[feat] = 20000000  # High throughput
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 250  # Moderate rate
        elif "IAT" in feat or "Inter Arrival" in feat:
            features[feat] = 4  # Regular intervals
        else:
            features[feat] = np.random.uniform(0, 300)
    
    return features


def test_attack_pattern(attack_name, features_func):
    """Test an attack pattern"""
    print(f"\n{'='*60}")
    print(f"Testing {attack_name} Attack Pattern")
    print(f"{'='*60}")
    
    features = features_func()
    if not features:
        print(f"❌ Could not create {attack_name} features")
        return None
    
    payload = {
        "features": features,
        "source_ip": "192.168.1.100",
        "attack_type": attack_name
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
            result = response.json()
            threat_score = result.get('threat_score', 0)
            threat_level = result.get('threat_level', 'Unknown')
            
            print(f"✅ Threat Score: {threat_score:.4f} ({threat_score*100:.2f}%)")
            print(f"🎯 Threat Level: {threat_level}")
            print(f"📊 Model Predictions: {json.dumps(result.get('model_predictions', {}), indent=2)}")
            
            if threat_score > 0.5:
                print(f"🚨 HIGH THREAT DETECTED!")
            elif threat_score > 0.2:
                print(f"⚠️  Moderate threat detected")
            else:
                print(f"✅ Low threat score")
            
            return result
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def main():
    """Run all IoT attack pattern tests"""
    print("=" * 60)
    print("IoT ATTACK PATTERN TESTS")
    print("=" * 60)
    print(f"API: {API_BASE_URL}")
    print(f"API Key: {API_KEY}")
    print()
    
    # Test all attack patterns
    attack_tests = [
        ("DoS", create_iot_dos_attack),
        ("DDoS", create_iot_ddos_attack),
        ("Mirai Botnet", create_mirai_botnet_attack),
        ("Brute Force", create_brute_force_attack),
        ("Recon", create_recon_attack),
        ("Spoofing", create_spoofing_attack),
        ("IIoT Attack", create_iiot_attack),
    ]
    
    results = []
    for attack_name, features_func in attack_tests:
        result = test_attack_pattern(attack_name, features_func)
        if result:
            results.append((attack_name, result.get('threat_score', 0)))
        import time
        time.sleep(0.5)  # Brief delay between requests
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for attack_name, threat_score in results:
        status = "🚨 HIGH" if threat_score > 0.5 else "⚠️  MODERATE" if threat_score > 0.2 else "✅ LOW"
        print(f"{status:12} | {attack_name:20} | Threat Score: {threat_score:.4f}")
    
    print(f"\n✅ Tested {len(results)}/{len(attack_tests)} attack patterns")
    print("🌐 Check your dashboard at http://localhost:3000 to see live data")


if __name__ == "__main__":
    main()

