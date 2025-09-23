#!/usr/bin/env python3
"""
Test script to generate realistic threat patterns for the AI cybersecurity tool
"""

import requests
import json
import numpy as np
import joblib
from pathlib import Path

def load_training_stats():
    """Load training data statistics to understand attack patterns"""
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    # Get attack samples (label=1)
    attack_samples = X_train[y_train == 1]
    benign_samples = X_train[y_train == 0]
    
    return {
        'attack_mean': attack_samples.mean(axis=0),
        'attack_std': attack_samples.std(axis=0),
        'attack_max': attack_samples.max(axis=0),
        'attack_min': attack_samples.min(axis=0),
        'benign_mean': benign_samples.mean(axis=0),
        'benign_std': benign_samples.std(axis=0)
    }

def generate_attack_pattern(stats, attack_type="ddos"):
    """Generate realistic attack pattern based on training data"""
    feature_names = joblib.load('models/feature_names.pkl')
    
    if attack_type == "ddos":
        # DDoS: High packet count, short duration, no backward packets
        pattern = stats['attack_mean'].copy()
        # Make it more extreme for DDoS
        pattern[2] = stats['attack_max'][2]  # Total Fwd Packets
        pattern[3] = stats['attack_min'][3]  # Total Backward Packets
        pattern[4] = stats['attack_max'][4]  # Total Length of Fwd Packets
        pattern[5] = stats['attack_min'][5]  # Total Length of Bwd Packets
        
    elif attack_type == "portscan":
        # Port scan: Many small packets, short duration
        pattern = stats['attack_mean'].copy()
        pattern[1] = stats['attack_min'][1]  # Flow Duration
        pattern[2] = stats['attack_max'][2]  # Total Fwd Packets
        pattern[3] = stats['attack_min'][3]  # Total Backward Packets
        
    elif attack_type == "botnet":
        # Botnet: Steady communication, medium duration
        pattern = stats['attack_mean'].copy()
        pattern[1] = stats['attack_mean'][1] + stats['attack_std'][1]  # Flow Duration
        
    else:
        # Use attack mean as baseline
        pattern = stats['attack_mean'].copy()
    
    # Convert to feature dictionary
    features = {}
    for i, feature_name in enumerate(feature_names):
        features[feature_name] = float(pattern[i])
    
    return features

def test_threat_detection():
    """Test the API with realistic threat patterns"""
    API_URL = "http://localhost:5001"
    
    print("🔍 Loading training data statistics...")
    stats = load_training_stats()
    
    print("📊 Training data analysis:")
    print(f"  Attack samples: {len(stats['attack_mean'])} features")
    print(f"  Attack mean range: {stats['attack_mean'].min():.3f} to {stats['attack_mean'].max():.3f}")
    print(f"  Attack std range: {stats['attack_std'].min():.3f} to {stats['attack_std'].max():.3f}")
    
    # Test different attack types
    attack_types = ["ddos", "portscan", "botnet", "baseline"]
    
    for attack_type in attack_types:
        print(f"\n🚨 Testing {attack_type.upper()} attack pattern...")
        
        features = generate_attack_pattern(stats, attack_type)
        features['source_ip'] = f"192.168.1.{np.random.randint(1, 255)}"
        features['attack_type'] = f"{attack_type}_test"
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                threat_score = result['threat_score']
                threat_level = result['threat_level']
                
                print(f"  ✅ Threat Score: {threat_score:.3f} ({threat_score*100:.1f}%)")
                print(f"  🎯 Threat Level: {threat_level}")
                print(f"  📈 Model Prediction: {result['model_predictions']}")
                
                if threat_score > 0.2:
                    print(f"  🚨 HIGH THREAT DETECTED!")
                elif threat_score > 0.1:
                    print(f"  ⚠️  Moderate threat detected")
                else:
                    print(f"  ✅ Low threat (normal behavior)")
                    
            else:
                print(f"  ❌ API Error: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
    
    # Get final stats
    print(f"\n📊 Final System Stats:")
    try:
        stats_response = requests.get(f"{API_URL}/stats", timeout=5)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            print(f"  Total Requests: {stats_data['total_requests']}")
            print(f"  Threats Detected: {stats_data['threats_detected']}")
            print(f"  Current Threat Level: {stats_data['current_threat_level']}")
            print(f"  Detection Rate: {(stats_data['threats_detected']/max(stats_data['total_requests'], 1)*100):.1f}%")
    except Exception as e:
        print(f"  ❌ Could not fetch stats: {e}")

if __name__ == "__main__":
    test_threat_detection()
