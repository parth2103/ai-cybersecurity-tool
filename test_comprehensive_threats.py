#!/usr/bin/env python3
"""
Comprehensive threat testing with all 78 features
"""

import requests
import json
import numpy as np
import joblib
import pandas as pd

def create_realistic_attack_pattern():
    """Create a realistic attack pattern with all 78 features"""
    feature_names = joblib.load('models/feature_names.pkl')
    
    # Start with realistic network values
    features = {}
    
    # Basic network flow features
    features['Destination Port'] = 80  # HTTP
    features['Flow Duration'] = 1000  # 1 second
    features['Total Fwd Packets'] = 10000  # High packet count
    features['Total Backward Packets'] = 0  # No response (DDoS pattern)
    features['Total Length of Fwd Packets'] = 15000000  # 15MB
    features['Total Length of Bwd Packets'] = 0  # No response
    
    # Packet length statistics
    features['Fwd Packet Length Max'] = 1500
    features['Fwd Packet Length Min'] = 1500
    features['Fwd Packet Length Mean'] = 1500
    features['Fwd Packet Length Std'] = 0
    features['Bwd Packet Length Max'] = 0
    features['Bwd Packet Length Min'] = 0
    features['Bwd Packet Length Mean'] = 0
    features['Bwd Packet Length Std'] = 0
    
    # Flow statistics
    features['Flow Bytes/s'] = 15000000  # 15MB/s
    features['Flow Packets/s'] = 10000  # 10k packets/s
    features['Flow IAT Mean'] = 0.1  # 100ms between packets
    features['Flow IAT Std'] = 0.01
    features['Flow IAT Max'] = 0.2
    features['Flow IAT Min'] = 0.05
    
    # Fill remaining features with realistic values
    for i, feature_name in enumerate(feature_names):
        if feature_name not in features:
            # Use realistic network values based on feature type
            if 'Fwd' in feature_name and 'Packet' in feature_name:
                features[feature_name] = 1500 if 'Max' in feature_name else 100
            elif 'Bwd' in feature_name and 'Packet' in feature_name:
                features[feature_name] = 0
            elif 'Active' in feature_name:
                features[feature_name] = 1000
            elif 'Idle' in feature_name:
                features[feature_name] = 100
            elif 'IAT' in feature_name:
                features[feature_name] = 0.1
            else:
                features[feature_name] = 100  # Default value
    
    return features

def test_extreme_attack_pattern():
    """Create an extreme attack pattern"""
    feature_names = joblib.load('models/feature_names.pkl')
    
    features = {}
    
    # Extreme DDoS pattern
    features['Destination Port'] = 80
    features['Flow Duration'] = 1  # 1ms
    features['Total Fwd Packets'] = 1000000  # 1M packets
    features['Total Backward Packets'] = 0
    features['Total Length of Fwd Packets'] = 1500000000  # 1.5GB
    features['Total Length of Bwd Packets'] = 0
    
    # All packets same size (botnet pattern)
    features['Fwd Packet Length Max'] = 1500
    features['Fwd Packet Length Min'] = 1500
    features['Fwd Packet Length Mean'] = 1500
    features['Fwd Packet Length Std'] = 0
    features['Bwd Packet Length Max'] = 0
    features['Bwd Packet Length Min'] = 0
    features['Bwd Packet Length Mean'] = 0
    features['Bwd Packet Length Std'] = 0
    
    # Extreme flow rates
    features['Flow Bytes/s'] = 1500000000  # 1.5GB/s
    features['Flow Packets/s'] = 1000000  # 1M packets/s
    features['Flow IAT Mean'] = 0.001  # 1ms
    features['Flow IAT Std'] = 0
    features['Flow IAT Max'] = 0.001
    features['Flow IAT Min'] = 0.001
    
    # Fill remaining with extreme values
    for i, feature_name in enumerate(feature_names):
        if feature_name not in features:
            if 'Fwd' in feature_name and 'Packet' in feature_name:
                features[feature_name] = 1500
            elif 'Bwd' in feature_name and 'Packet' in feature_name:
                features[feature_name] = 0
            elif 'Active' in feature_name:
                features[feature_name] = 1000000
            elif 'Idle' in feature_name:
                features[feature_name] = 0
            elif 'IAT' in feature_name:
                features[feature_name] = 0.001
            else:
                features[feature_name] = 1000000
    
    return features

def test_threat_detection():
    """Test comprehensive threat detection"""
    API_URL = "http://localhost:5001"
    
    print("🚨 Testing Comprehensive Threat Detection")
    print("=" * 50)
    
    # Test 1: Realistic attack pattern
    print("\n1️⃣ Testing Realistic DDoS Attack Pattern...")
    features1 = create_realistic_attack_pattern()
    features1['source_ip'] = "192.168.1.100"
    features1['attack_type'] = "DDoS_Realistic"
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features1},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            threat_score = result['threat_score']
            threat_level = result['threat_level']
            
            print(f"   ✅ Threat Score: {threat_score:.3f} ({threat_score*100:.1f}%)")
            print(f"   🎯 Threat Level: {threat_level}")
            
            if threat_score > 0.2:
                print(f"   🚨 HIGH THREAT DETECTED!")
            elif threat_score > 0.1:
                print(f"   ⚠️  Moderate threat detected")
            else:
                print(f"   ✅ Low threat (normal behavior)")
        else:
            print(f"   ❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 2: Extreme attack pattern
    print("\n2️⃣ Testing Extreme DDoS Attack Pattern...")
    features2 = test_extreme_attack_pattern()
    features2['source_ip'] = "10.0.0.1"
    features2['attack_type'] = "DDoS_Extreme"
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features2},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            threat_score = result['threat_score']
            threat_level = result['threat_level']
            
            print(f"   ✅ Threat Score: {threat_score:.3f} ({threat_score*100:.1f}%)")
            print(f"   🎯 Threat Level: {threat_level}")
            
            if threat_score > 0.2:
                print(f"   🚨 HIGH THREAT DETECTED!")
            elif threat_score > 0.1:
                print(f"   ⚠️  Moderate threat detected")
            else:
                print(f"   ✅ Low threat (normal behavior)")
        else:
            print(f"   ❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 3: Port scan pattern
    print("\n3️⃣ Testing Port Scan Attack Pattern...")
    features3 = create_realistic_attack_pattern()
    features3['Destination Port'] = 22  # SSH
    features3['Flow Duration'] = 100  # Very short
    features3['Total Fwd Packets'] = 1  # Single packet
    features3['Total Length of Fwd Packets'] = 60  # Small packet
    features3['Flow Bytes/s'] = 600
    features3['Flow Packets/s'] = 10
    features3['source_ip'] = "192.168.1.200"
    features3['attack_type'] = "Port_Scan"
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features3},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            threat_score = result['threat_score']
            threat_level = result['threat_level']
            
            print(f"   ✅ Threat Score: {threat_score:.3f} ({threat_score*100:.1f}%)")
            print(f"   🎯 Threat Level: {threat_level}")
            
            if threat_score > 0.2:
                print(f"   🚨 HIGH THREAT DETECTED!")
            elif threat_score > 0.1:
                print(f"   ⚠️  Moderate threat detected")
            else:
                print(f"   ✅ Low threat (normal behavior)")
        else:
            print(f"   ❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
    
    # Get final stats
    print(f"\n📊 Final System Stats:")
    try:
        stats_response = requests.get(f"{API_URL}/stats", timeout=5)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            print(f"   Total Requests: {stats_data['total_requests']}")
            print(f"   Threats Detected: {stats_data['threats_detected']}")
            print(f"   Current Threat Level: {stats_data['current_threat_level']}")
            print(f"   Detection Rate: {(stats_data['threats_detected']/max(stats_data['total_requests'], 1)*100):.1f}%")
            
            # Show recent threat history
            if stats_data['threat_history']:
                print(f"\n   Recent Threat Scores:")
                for i, entry in enumerate(stats_data['threat_history'][-5:], 1):
                    print(f"     {i}. {entry['threat_score']:.3f} ({entry['threat_level']}) - {entry['timestamp'][:19]}")
                    
    except Exception as e:
        print(f"   ❌ Could not fetch stats: {e}")

if __name__ == "__main__":
    test_threat_detection()
