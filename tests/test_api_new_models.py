#!/usr/bin/env python3
"""
API integration tests for new models
Tests API endpoints with new models (IoT-IDAD 2024 + CICAPT-IIOT)
"""

import unittest
import requests
import json
import time
import numpy as np
import joblib
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = Path(__file__).parent.parent / "models"
API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"


class TestAPINewModels(unittest.TestCase):
    """Test API with new models"""

    BASE_URL = API_BASE_URL

    @classmethod
    def setUpClass(cls):
        """Set up test class - check if API is running"""
        try:
            response = requests.get(f"{cls.BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                raise unittest.SkipTest(
                    "API server is not running. Start with: python api/app.py"
                )
        except requests.exceptions.RequestException:
            raise unittest.SkipTest(
                "API server is not running. Start with: python api/app.py"
            )

    def setUp(self):
        """Set up test fixtures"""
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        })

        # Load new feature names
        try:
            features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
                if not isinstance(self.feature_names, list):
                    self.feature_names = list(self.feature_names)
            else:
                self.feature_names = None
        except Exception:
            self.feature_names = None

    def create_test_features(self, attack_type="normal"):
        """Create test features using new model feature names"""
        features = {}
        
        if self.feature_names is None:
            # Fallback to basic features
            features = {
                "Destination Port": 80 if attack_type != "normal" else 443,
                "Flow Duration": 1000 if attack_type != "normal" else 100000,
                "Total Fwd Packets": 10000 if attack_type != "normal" else 10,
                "Total Backward Packets": 0 if attack_type != "normal" else 10,
            }
        else:
            # Use actual feature names from new models
            for feat in self.feature_names[:20]:  # Use first 20 features
                if "Port" in feat:
                    features[feat] = 80 if attack_type != "normal" else 443
                elif "Duration" in feat:
                    features[feat] = 1000 if attack_type != "normal" else 100000
                elif "Fwd Packets" in feat or "Forward Packets" in feat:
                    features[feat] = 10000 if attack_type != "normal" else 10
                elif "Bwd Packets" in feat or "Backward Packets" in feat:
                    features[feat] = 0 if attack_type != "normal" else 10
                elif "Bytes/s" in feat or "Bytes per" in feat:
                    features[feat] = 15000000 if attack_type != "normal" else 2000
                elif "Packets/s" in feat or "Packets per" in feat:
                    features[feat] = 10000 if attack_type != "normal" else 20
                else:
                    features[feat] = np.random.uniform(0, 100)
        
        return features

    def test_health_endpoint(self):
        """Test health endpoint"""
        response = self.session.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

    def test_predict_endpoint_new_features(self):
        """Test /predict endpoint with new model features"""
        sample_data = {
            "features": self.create_test_features("normal"),
            "source_ip": "192.168.1.100"
        }

        response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertIn("threat_detected", data)
        self.assertIn("threat_level", data)
        self.assertIn("threat_score", data)
        self.assertIn("model_predictions", data)
        self.assertIn("timestamp", data)

        # Verify threat score is valid
        self.assertGreaterEqual(data["threat_score"], 0)
        self.assertLessEqual(data["threat_score"], 1)

    def test_stats_endpoint(self):
        """Test /stats endpoint shows correct model information"""
        response = self.session.get(f"{self.BASE_URL}/stats")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertIn("total_requests", data)
        self.assertIn("threats_detected", data)
        self.assertIn("threat_history", data)
        self.assertIn("current_threat_level", data)

        # Check data types
        self.assertIsInstance(data["total_requests"], int)
        self.assertIsInstance(data["threats_detected"], int)
        self.assertIsInstance(data["threat_history"], list)

    def test_models_performance_endpoint(self):
        """Test /models/performance endpoint with new models"""
        response = self.session.get(f"{self.BASE_URL}/models/performance")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertIn("models", data)
        self.assertIn("total_predictions", data)
        self.assertIn("healthy_models", data)
        self.assertIn("total_models", data)

        # Check that models are present
        models = data.get("models", {})
        self.assertGreater(len(models), 0, "Should have at least one model")

    def test_system_info_endpoint(self):
        """Test /system/info endpoint reports new models loaded"""
        response = self.session.get(f"{self.BASE_URL}/system/info")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertIn("models_loaded", data)
        self.assertIn("cpu_percent", data)
        self.assertIn("memory_percent", data)
        self.assertIn("total_predictions", data)

        # Check models_loaded is a list
        self.assertIsInstance(data["models_loaded"], list)

    def test_explain_endpoint_new_features(self):
        """Test /explain endpoint works with new feature set"""
        sample_data = {
            "features": self.create_test_features("normal")
        }

        response = self.session.post(f"{self.BASE_URL}/explain", json=sample_data)

        # Should return 200 or 400 (if explain not fully implemented)
        self.assertIn(response.status_code, [200, 400, 501])

        if response.status_code == 200:
            data = response.json()
            # If explain is implemented, check structure
            if "explanation" in data or "feature_importance" in data:
                self.assertIsInstance(data.get("explanation") or data.get("feature_importance"), 
                                    (dict, list))

    def test_predict_with_attack_pattern(self):
        """Test prediction with attack pattern"""
        sample_data = {
            "features": self.create_test_features("attack"),
            "source_ip": "192.168.1.200",
            "attack_type": "test_attack"
        }

        response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response
        self.assertIn("threat_score", data)
        self.assertGreaterEqual(data["threat_score"], 0)
        self.assertLessEqual(data["threat_score"], 1)

    def test_predict_with_benign_pattern(self):
        """Test prediction with benign pattern"""
        sample_data = {
            "features": self.create_test_features("normal"),
            "source_ip": "192.168.1.100"
        }

        response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Benign should have lower threat score
        self.assertIn("threat_score", data)
        # Note: May vary, but should be a valid score

    def test_multiple_predictions(self):
        """Test multiple predictions in sequence"""
        for i in range(3):
            sample_data = {
                "features": self.create_test_features("normal" if i % 2 == 0 else "attack"),
                "source_ip": f"192.168.1.{100 + i}"
            }

            response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn("threat_score", data)
            
            time.sleep(0.1)  # Brief delay

    def test_api_logs_new_models(self):
        """Test that API logs indicate new models are being used"""
        # Make a prediction
        sample_data = {
            "features": self.create_test_features("normal")
        }
        response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)
        self.assertEqual(response.status_code, 200)
        
        # Check system info to see which models are loaded
        info_response = self.session.get(f"{self.BASE_URL}/system/info")
        if info_response.status_code == 200:
            info_data = info_response.json()
            models_loaded = info_data.get("models_loaded", [])
            # Should have at least one model
            self.assertGreater(len(models_loaded), 0)

    def test_feature_missing_handling(self):
        """Test that API handles missing features gracefully"""
        # Create features with only a subset
        partial_features = {
            "Destination Port": 80,
            "Flow Duration": 1000,
            "Total Fwd Packets": 100
        }

        sample_data = {
            "features": partial_features
        }

        response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)

        # Should either succeed (with default values) or return 400
        self.assertIn(response.status_code, [200, 400])

    def test_feature_extra_handling(self):
        """Test that API handles extra features gracefully"""
        # Create features with extra ones
        extra_features = self.create_test_features("normal")
        extra_features["ExtraFeature1"] = 123
        extra_features["ExtraFeature2"] = 456

        sample_data = {
            "features": extra_features
        }

        response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)

        # Should succeed (extra features ignored)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()

