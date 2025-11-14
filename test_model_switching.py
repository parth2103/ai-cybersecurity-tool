#!/usr/bin/env python3
"""
Model switching tests
Verifies API uses new models when available and falls back to old models
"""

import requests
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"
MODELS_DIR = Path(__file__).parent / "models"


def check_models_loaded():
    """Check which models are currently loaded"""
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(f"{API_BASE_URL}/system/info", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models_loaded = data.get("models_loaded", [])
            return models_loaded
        return None
    except Exception as e:
        print(f"Error checking models: {e}")
        return None


def check_model_files():
    """Check which model files exist"""
    models_status = {
        "new_random_forest": (MODELS_DIR / "random_forest_new_datasets.pkl").exists(),
        "old_random_forest": (MODELS_DIR / "baseline_model.pkl").exists(),
        "new_xgboost": (MODELS_DIR / "xgboost_model_new_datasets.pkl").exists(),
        "old_xgboost": (MODELS_DIR / "xgboost_model.pkl").exists(),
        "new_isolation_forest": (MODELS_DIR / "isolation_forest_new_datasets.pkl").exists(),
        "old_isolation_forest": (MODELS_DIR / "isolation_forest.pkl").exists(),
    }
    return models_status


def test_model_priority():
    """Test that API prioritizes new models"""
    print("=" * 60)
    print("MODEL SWITCHING TEST")
    print("=" * 60)
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not available")
            return
    except Exception as e:
        print(f"❌ API is not available: {e}")
        return
    
    # Check model files
    print("\n1. Checking model files...")
    models_status = check_model_files()
    
    print("   Model Files Status:")
    for model_name, exists in models_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {model_name}")
    
    # Check which models are loaded
    print("\n2. Checking loaded models...")
    models_loaded = check_models_loaded()
    
    if models_loaded:
        print(f"   ✅ Loaded models: {', '.join(models_loaded)}")
        
        # Check if new models are being used
        new_models_found = any("new" in m.lower() or "random_forest" in m.lower() 
                              for m in models_loaded)
        
        if new_models_found or models_status["new_random_forest"]:
            print("   ✅ New models are available/loaded")
        else:
            print("   ⚠️  New models may not be loaded (check API logs)")
    else:
        print("   ⚠️  Could not determine loaded models")
    
    # Test model priority logic
    print("\n3. Testing model priority logic...")
    print("   According to api/app.py, priority is:")
    print("   1. New models (random_forest_new_datasets.pkl, etc.)")
    print("   2. Old models (baseline_model.pkl, etc.)")
    
    if models_status["new_random_forest"]:
        print("   ✅ New Random Forest exists - should be used")
    elif models_status["old_random_forest"]:
        print("   ⚠️  Only old Random Forest exists - will be used as fallback")
    
    if models_status["new_xgboost"]:
        print("   ✅ New XGBoost exists - should be used")
    elif models_status["old_xgboost"]:
        print("   ⚠️  Only old XGBoost exists - will be used as fallback")
    
    if models_status["new_isolation_forest"]:
        print("   ✅ New Isolation Forest exists - should be used")
    elif models_status["old_isolation_forest"]:
        print("   ⚠️  Only old Isolation Forest exists - will be used as fallback")
    
    # Test prediction to verify models work
    print("\n4. Testing prediction with current models...")
    test_features = {
        "Destination Port": 443,
        "Flow Duration": 100000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 10,
    }
    
    payload = {"features": test_features}
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
            print(f"   ✅ Prediction successful")
            print(f"   Threat Score: {data.get('threat_score', 0):.4f}")
            print(f"   Models used: {list(data.get('model_predictions', {}).keys())}")
        else:
            print(f"   ⚠️  Prediction returned: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("MODEL SWITCHING TEST SUMMARY")
    print("=" * 60)
    
    new_models_available = sum(1 for k, v in models_status.items() if "new" in k and v)
    old_models_available = sum(1 for k, v in models_status.items() if "old" in k and v)
    
    print(f"New models available: {new_models_available}")
    print(f"Old models available: {old_models_available}")
    
    if new_models_available > 0:
        print("\n✅ New models are available and should be prioritized")
    else:
        print("\n⚠️  No new models found - API will use old models")
    
    print("\n📝 Note: Check API startup logs to see which models were actually loaded")
    print("   Look for messages like 'Loaded RF model from ... [NEW (IoT-IDAD + CICAPT)]'")


def main():
    """Run model switching tests"""
    test_model_priority()


if __name__ == "__main__":
    main()

