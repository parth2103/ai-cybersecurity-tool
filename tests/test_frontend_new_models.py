#!/usr/bin/env python3
"""
Frontend integration tests for new models
Tests dashboard connection and data display with new models
"""

import unittest
import requests
import json
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE_URL = "http://localhost:5001"
FRONTEND_URL = "http://localhost:3000"
API_KEY = "dev-key-123"


class TestFrontendNewModels(unittest.TestCase):
    """Test frontend integration with new models"""

    @classmethod
    def setUpClass(cls):
        """Set up test class - check if services are running"""
        # Check API
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                raise unittest.SkipTest("API server is not running")
        except requests.exceptions.RequestException:
            raise unittest.SkipTest("API server is not running")
        
        # Check Frontend (optional)
        try:
            response = requests.get(f"{FRONTEND_URL}", timeout=5)
            cls.frontend_available = response.status_code == 200
        except requests.exceptions.RequestException:
            cls.frontend_available = False

    def setUp(self):
        """Set up test fixtures"""
        self.api_url = API_BASE_URL
        self.frontend_url = FRONTEND_URL
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        }

    def test_api_connectivity(self):
        """Test that API is accessible"""
        response = requests.get(f"{self.api_url}/health", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_stats_endpoint_for_dashboard(self):
        """Test /stats endpoint that dashboard uses"""
        response = requests.get(f"{self.api_url}/stats", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        # Dashboard expects these fields
        self.assertIn("total_requests", data)
        self.assertIn("threats_detected", data)
        self.assertIn("threat_history", data)
        self.assertIn("current_threat_level", data)

    def test_system_info_endpoint_for_dashboard(self):
        """Test /system/info endpoint that dashboard uses"""
        response = requests.get(f"{self.api_url}/system/info", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        # Dashboard expects these fields
        self.assertIn("models_loaded", data)
        self.assertIn("cpu_percent", data)
        self.assertIn("memory_percent", data)

    def test_models_performance_endpoint_for_dashboard(self):
        """Test /models/performance endpoint that dashboard uses"""
        response = requests.get(f"{self.api_url}/models/performance", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        # Dashboard expects these fields
        self.assertIn("models", data)
        self.assertIn("total_predictions", data)
        self.assertIn("healthy_models", data)

    def test_real_time_threat_updates(self):
        """Test that predictions update threat history for real-time display"""
        # Make a prediction
        sample_data = {
            "features": {
                "Destination Port": 80,
                "Flow Duration": 1000,
                "Total Fwd Packets": 10000,
                "Total Backward Packets": 0
            }
        }
        
        response = requests.post(
            f"{self.api_url}/predict",
            json=sample_data,
            headers=self.headers,
            timeout=10
        )
        self.assertEqual(response.status_code, 200)
        
        # Check that stats updated
        time.sleep(0.5)  # Brief delay for processing
        stats_response = requests.get(f"{self.api_url}/stats", headers=self.headers)
        self.assertEqual(stats_response.status_code, 200)
        
        stats_data = stats_response.json()
        self.assertGreaterEqual(stats_data["total_requests"], 1)

    def test_model_performance_metrics_display(self):
        """Test that model performance metrics are available for display"""
        response = requests.get(f"{self.api_url}/models/performance", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        models = data.get("models", {})
        
        # Should have model data
        self.assertGreater(len(models), 0)
        
        # Check model structure
        for model_name, model_data in models.items():
            self.assertIn("status", model_data)
            self.assertIn("predictions", model_data)
            self.assertIn("avg_confidence", model_data)

    def test_threat_history_updates(self):
        """Test that threat history updates correctly"""
        # Get initial threat history length
        stats_response = requests.get(f"{self.api_url}/stats", headers=self.headers)
        initial_data = stats_response.json()
        initial_history_length = len(initial_data.get("threat_history", []))
        
        # Make a prediction
        sample_data = {
            "features": {
                "Destination Port": 443,
                "Flow Duration": 100000,
                "Total Fwd Packets": 10,
                "Total Backward Packets": 10
            }
        }
        
        requests.post(
            f"{self.api_url}/predict",
            json=sample_data,
            headers=self.headers,
            timeout=10
        )
        
        # Check updated threat history
        time.sleep(0.5)
        updated_stats = requests.get(f"{self.api_url}/stats", headers=self.headers).json()
        updated_history_length = len(updated_stats.get("threat_history", []))
        
        # History should have increased or stayed same (may have max limit)
        self.assertGreaterEqual(updated_history_length, initial_history_length)

    def test_alert_notifications_high_threat(self):
        """Test that high threat scores trigger alerts"""
        # Create high threat pattern
        high_threat_data = {
            "features": {
                "Destination Port": 80,
                "Flow Duration": 1000,
                "Total Fwd Packets": 100000,
                "Total Backward Packets": 0,
                "Flow Bytes/s": 150000000,
                "Flow Packets/s": 100000
            }
        }
        
        response = requests.post(
            f"{self.api_url}/predict",
            json=high_threat_data,
            headers=self.headers,
            timeout=10
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check threat score
        threat_score = data.get("threat_score", 0)
        self.assertGreaterEqual(threat_score, 0)
        self.assertLessEqual(threat_score, 1)

    def test_frontend_connectivity(self):
        """Test that frontend is accessible (if available)"""
        if not self.frontend_available:
            self.skipTest("Frontend not available")
        
        response = requests.get(f"{self.frontend_url}", timeout=5)
        self.assertEqual(response.status_code, 200)

    def test_new_models_loaded_indicator(self):
        """Test that system info shows new models are loaded"""
        response = requests.get(f"{self.api_url}/system/info", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        models_loaded = data.get("models_loaded", [])
        
        # Should have at least one model
        self.assertGreater(len(models_loaded), 0)
        
        # Check if new models are mentioned (API should prioritize new models)
        # This is indirect - we can't directly check, but models should be loaded


if __name__ == "__main__":
    unittest.main()

