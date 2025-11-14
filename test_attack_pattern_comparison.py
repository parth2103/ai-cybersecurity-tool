#!/usr/bin/env python3
"""
Compare attack pattern detection: old models vs new models
Tests same attack pattern on both model sets
"""

import requests
import json
import numpy as np
import joblib
from pathlib import Path

API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"
MODELS_DIR = Path(__file__).parent / "models"


def create_test_attack_pattern():
    """Create a test attack pattern"""
    # Try to load new feature names
    features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
    if features_path.exists():
        feature_names = joblib.load(features_path)
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)
    else:
        # Fallback to basic features
        feature_names = [
            "Destination Port", "Flow Duration", "Total Fwd Packets",
            "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s"
        ]
    
    features = {}
    for feat in feature_names:
        if "Port" in feat:
            features[feat] = 80
        elif "Duration" in feat:
            features[feat] = 1000
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            features[feat] = 10000
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            features[feat] = 0
        elif "Bytes/s" in feat or "Bytes per" in feat:
            features[feat] = 15000000
        elif "Packets/s" in feat or "Packets per" in feat:
            features[feat] = 10000
        else:
            features[feat] = np.random.uniform(0, 100)
    
    return features


def test_with_current_models(attack_pattern, pattern_name):
    """Test attack pattern with currently loaded models (new models)"""
    payload = {
        "features": attack_pattern,
        "source_ip": "192.168.1.100",
        "attack_type": pattern_name
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
            return {
                "threat_score": result.get('threat_score', 0),
                "threat_level": result.get('threat_level', 'Unknown'),
                "model_predictions": result.get('model_predictions', {})
            }
        else:
            return None
    except Exception as e:
        print(f"Error testing with current models: {e}")
        return None


def compare_detection_rates():
    """Compare detection rates between old and new models"""
    print("=" * 60)
    print("ATTACK PATTERN COMPARISON: OLD vs NEW MODELS")
    print("=" * 60)
    print(f"API: {API_BASE_URL}")
    print()
    
    # Note: We can't directly test old models if API is using new models
    # But we can test the same pattern and note which models are being used
    print("Note: API is configured to use new models by default")
    print("Testing with currently active models (should be new models)...")
    print()
    
    # Create test patterns
    test_patterns = [
        ("DDoS Attack", create_test_attack_pattern()),
        ("DoS Attack", create_test_attack_pattern()),
        ("Port Scan", create_test_attack_pattern()),
    ]
    
    results = []
    for pattern_name, pattern_features in test_patterns:
        print(f"\nTesting: {pattern_name}")
        print("-" * 60)
        
        result = test_with_current_models(pattern_features, pattern_name)
        if result:
            threat_score = result['threat_score']
            threat_level = result['threat_level']
            
            print(f"Threat Score: {threat_score:.4f} ({threat_score*100:.2f}%)")
            print(f"Threat Level: {threat_level}")
            
            results.append({
                "pattern": pattern_name,
                "threat_score": threat_score,
                "threat_level": threat_level
            })
        else:
            print("❌ Failed to get prediction")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if results:
        avg_threat_score = np.mean([r['threat_score'] for r in results])
        high_threat_count = sum(1 for r in results if r['threat_score'] > 0.5)
        moderate_threat_count = sum(1 for r in results if 0.2 < r['threat_score'] <= 0.5)
        
        print(f"Average Threat Score: {avg_threat_score:.4f}")
        print(f"High Threat Detections: {high_threat_count}/{len(results)}")
        print(f"Moderate Threat Detections: {moderate_threat_count}/{len(results)}")
        print(f"Low Threat Detections: {len(results) - high_threat_count - moderate_threat_count}/{len(results)}")
        
        print("\nPer-Pattern Results:")
        for r in results:
            status = "🚨 HIGH" if r['threat_score'] > 0.5 else "⚠️  MODERATE" if r['threat_score'] > 0.2 else "✅ LOW"
            print(f"  {status:12} | {r['pattern']:20} | Score: {r['threat_score']:.4f}")
    
    print("\n" + "=" * 60)
    print("NOTE: To compare with old models, you would need to:")
    print("1. Temporarily switch API to use old models")
    print("2. Run the same test patterns")
    print("3. Compare the threat scores")
    print("=" * 60)


def test_false_positive_rate():
    """Test false positive rate with benign traffic"""
    print("\n" + "=" * 60)
    print("FALSE POSITIVE RATE TEST (Benign Traffic)")
    print("=" * 60)
    
    # Create benign traffic pattern
    features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
    if features_path.exists():
        feature_names = joblib.load(features_path)
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)
    else:
        feature_names = ["Destination Port", "Flow Duration", "Total Fwd Packets"]
    
    benign_features = {}
    for feat in feature_names:
        if "Port" in feat:
            benign_features[feat] = 443  # HTTPS
        elif "Duration" in feat:
            benign_features[feat] = 100000  # Normal duration
        elif "Fwd Packets" in feat or "Forward Packets" in feat:
            benign_features[feat] = 10  # Normal packet count
        elif "Bwd Packets" in feat or "Backward Packets" in feat:
            benign_features[feat] = 10  # Normal response
        elif "Bytes/s" in feat or "Bytes per" in feat:
            benign_features[feat] = 2000  # Normal throughput
        elif "Packets/s" in feat or "Packets per" in feat:
            benign_features[feat] = 20  # Normal rate
        else:
            benign_features[feat] = np.random.uniform(0, 100)
    
    # Test multiple benign samples
    false_positives = 0
    total_tests = 10
    
    print(f"Testing {total_tests} benign traffic samples...")
    for i in range(total_tests):
        result = test_with_current_models(benign_features, f"Benign_{i}")
        if result:
            threat_score = result['threat_score']
            if threat_score > 0.2:  # Consider > 20% as false positive
                false_positives += 1
                print(f"  Sample {i+1}: Threat Score {threat_score:.4f} (FALSE POSITIVE)")
            else:
                print(f"  Sample {i+1}: Threat Score {threat_score:.4f} (OK)")
    
    false_positive_rate = false_positives / total_tests
    print(f"\nFalse Positive Rate: {false_positive_rate:.2%} ({false_positives}/{total_tests})")
    
    if false_positive_rate < 0.1:
        print("✅ Good: False positive rate < 10%")
    elif false_positive_rate < 0.2:
        print("⚠️  Acceptable: False positive rate < 20%")
    else:
        print("❌ High: False positive rate >= 20%")


def main():
    """Run comparison tests"""
    try:
        # Check API health
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not available at {API_BASE_URL}")
            print("   Start API with: python api/app.py")
            return
    except Exception as e:
        print(f"❌ API not available: {e}")
        return
    
    # Run comparison
    compare_detection_rates()
    
    # Test false positive rate
    test_false_positive_rate()
    
    print("\n✅ Comparison tests complete!")
    print("🌐 Check your dashboard at http://localhost:3000")


if __name__ == "__main__":
    main()

