#!/usr/bin/env python3
"""
Integration tests for the Attention Explainability API
Tests the /explain endpoint with real network features
"""
import requests
import json
import numpy as np
import time

API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

def test_health_check():
    """Test API health check"""
    print("\n[1/6] Testing API health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        print("✓ API is healthy")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_explain_health():
    """Test explain endpoint health"""
    print("\n[2/6] Testing explain endpoint health...")
    try:
        response = requests.get(f"{API_BASE_URL}/explain/health", headers=HEADERS, timeout=5)
        data = response.json()
        print(f"  Explainer status: {data.get('status', 'unknown')}")
        print(f"  Details: {json.dumps(data.get('details', {}), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Explain health check failed: {e}")
        return False

def test_explain_single():
    """Test single prediction explanation"""
    print("\n[3/6] Testing single prediction explanation...")

    # Sample network features (must match model's expected features)
    features = {
        "Feature_1": 0.5,
        "Feature_2": 0.8,
        "Feature_3": 0.3,
    }

    payload = {
        "features": features,
        "prediction": 0.85
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/explain",
            json=payload,
            headers=HEADERS,
            timeout=10
        )

        print(f"  Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  Success: {data.get('success', False)}")

            if 'explanation' in data:
                exp = data['explanation']
                print(f"  Threat Level: {exp.get('threat_level', 'N/A')}")
                print(f"  Prediction: {exp.get('prediction', 'N/A')}")
                print(f"\n  Top Features:")
                for feature, score in exp.get('top_features', [])[:3]:
                    print(f"    - {feature}: {score:.4f}")

                print(f"\n  Explanation Text:\n{exp.get('explanation', 'N/A')}")
                print("✓ Single explanation test passed")
                return True
        else:
            print(f"✗ Request failed: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Single explanation test failed: {e}")
        return False

def test_explain_batch():
    """Test batch prediction explanations"""
    print("\n[4/6] Testing batch prediction explanations...")

    batch = [
        {
            "features": {
                "Feature_1": 0.3,
                "Feature_2": 0.4,
                "Feature_3": 0.2,
            },
            "prediction": 0.2
        },
        {
            "features": {
                "Feature_1": 0.9,
                "Feature_2": 0.85,
                "Feature_3": 0.75,
            },
            "prediction": 0.9
        }
    ]

    payload = {"batch": batch}

    try:
        response = requests.post(
            f"{API_BASE_URL}/explain/batch",
            json=payload,
            headers=HEADERS,
            timeout=10
        )

        print(f"  Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  Success: {data.get('success', False)}")
            print(f"  Explanations count: {data.get('count', 0)}")

            for i, exp in enumerate(data.get('explanations', [])):
                print(f"\n  Explanation {i+1}:")
                print(f"    Threat Level: {exp.get('threat_level', 'N/A')}")
                print(f"    Prediction: {exp.get('prediction', 'N/A')}")

            print("✓ Batch explanation test passed")
            return True
        else:
            print(f"✗ Request failed: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Batch explanation test failed: {e}")
        return False

def test_visualization_data():
    """Test visualization data format"""
    print("\n[5/6] Testing visualization data format...")

    features = {
        "Feature_1": 0.6,
        "Feature_2": 0.7,
        "Feature_3": 0.4,
    }

    payload = {
        "features": features,
        "prediction": 0.75
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/explain",
            json=payload,
            headers=HEADERS,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            if 'visualization_data' in data and data['visualization_data']:
                viz = data['visualization_data']
                print(f"  Features count: {len(viz.get('features', []))}")
                print(f"  Weights count: {len(viz.get('weights', []))}")

                assert len(viz['features']) == len(viz['weights'])
                assert all(isinstance(w, (int, float)) for w in viz['weights'])

                print("  Top visualization features:")
                for feat, weight in zip(viz['features'][:5], viz['weights'][:5]):
                    print(f"    - {feat}: {weight:.4f}")

                print("✓ Visualization data test passed")
                return True
        else:
            print(f"✗ Request failed: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid data"""
    print("\n[6/6] Testing error handling...")

    test_cases = [
        ("Empty features", {}),
        ("Invalid features", {"features": "invalid"}),
        ("Missing features", {"prediction": 0.5}),
    ]

    passed = 0
    for test_name, payload in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/explain",
                json=payload,
                headers=HEADERS,
                timeout=5
            )

            # Should get an error response
            if response.status_code in [400, 500]:
                print(f"  ✓ {test_name}: Correctly handled (status {response.status_code})")
                passed += 1
            else:
                print(f"  ✗ {test_name}: Unexpected status {response.status_code}")
        except Exception as e:
            print(f"  ✗ {test_name}: {e}")

    if passed == len(test_cases):
        print("✓ All error handling tests passed")
        return True
    else:
        print(f"✗ {passed}/{len(test_cases)} error handling tests passed")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ATTENTION EXPLAINABILITY API INTEGRATION TESTS")
    print("=" * 60)
    print(f"API URL: {API_BASE_URL}")
    print(f"API Key: {API_KEY}")

    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))

    if results[0][1]:  # Only continue if API is healthy
        results.append(("Explain Health", test_explain_health()))
        results.append(("Single Explanation", test_explain_single()))
        results.append(("Batch Explanation", test_explain_batch()))
        results.append(("Visualization Data", test_visualization_data()))
        results.append(("Error Handling", test_error_handling()))
    else:
        print("\n⚠️  API is not available. Skipping remaining tests.")
        print("   Make sure the API server is running on port 5001")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Attention explainability is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
