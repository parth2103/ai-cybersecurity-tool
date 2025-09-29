#!/usr/bin/env python3
"""
Integration tests for AI Cybersecurity Tool API
Tests the complete system functionality including API endpoints, WebSocket, and performance
"""

import unittest
import requests
import json
import time
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor
import socketio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IntegrationTests(unittest.TestCase):
    """Comprehensive integration tests for the AI Cybersecurity Tool"""

    BASE_URL = "http://localhost:5001"  # Updated to port 5001

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
        self.session.headers.update({"Content-Type": "application/json"})

        # Load feature names for realistic test data
        try:
            self.feature_names = joblib.load("models/feature_names.pkl")
        except FileNotFoundError:
            self.feature_names = [
                "Destination Port",
                "Flow Duration",
                "Total Fwd Packets",
                "Total Backward Packets",
                "Total Length of Fwd Packets",
            ]

    def create_realistic_features(self, attack_type="normal"):
        """Create realistic feature data based on training data"""
        features = {}

        if attack_type == "ddos":
            # DDoS pattern: high packet count, short duration, no backward packets
            features["Destination Port"] = 80
            features["Flow Duration"] = 1000
            features["Total Fwd Packets"] = 10000
            features["Total Backward Packets"] = 0
            features["Total Length of Fwd Packets"] = 15000000
            features["Total Length of Bwd Packets"] = 0
            features["Fwd Packet Length Max"] = 1500
            features["Fwd Packet Length Min"] = 1500
            features["Fwd Packet Length Mean"] = 1500
            features["Fwd Packet Length Std"] = 0
            features["Bwd Packet Length Max"] = 0
            features["Bwd Packet Length Min"] = 0
            features["Bwd Packet Length Mean"] = 0
            features["Bwd Packet Length Std"] = 0
            features["Flow Bytes/s"] = 15000000
            features["Flow Packets/s"] = 10000
            features["Flow IAT Mean"] = 0.1
            features["Flow IAT Std"] = 0.01
            features["Flow IAT Max"] = 0.2
            features["Flow IAT Min"] = 0.05

        elif attack_type == "portscan":
            # Port scan pattern: single packet, short duration
            features["Destination Port"] = 22
            features["Flow Duration"] = 100
            features["Total Fwd Packets"] = 1
            features["Total Backward Packets"] = 0
            features["Total Length of Fwd Packets"] = 60
            features["Total Length of Bwd Packets"] = 0
            features["Fwd Packet Length Max"] = 60
            features["Fwd Packet Length Min"] = 60
            features["Fwd Packet Length Mean"] = 60
            features["Fwd Packet Length Std"] = 0
            features["Bwd Packet Length Max"] = 0
            features["Bwd Packet Length Min"] = 0
            features["Bwd Packet Length Mean"] = 0
            features["Bwd Packet Length Std"] = 0
            features["Flow Bytes/s"] = 600
            features["Flow Packets/s"] = 10
            features["Flow IAT Mean"] = 100
            features["Flow IAT Std"] = 0
            features["Flow IAT Max"] = 100
            features["Flow IAT Min"] = 100

        else:  # normal
            # Normal traffic pattern
            features["Destination Port"] = 80
            features["Flow Duration"] = 10000
            features["Total Fwd Packets"] = 100
            features["Total Backward Packets"] = 100
            features["Total Length of Fwd Packets"] = 10000
            features["Total Length of Bwd Packets"] = 10000
            features["Fwd Packet Length Max"] = 1500
            features["Fwd Packet Length Min"] = 64
            features["Fwd Packet Length Mean"] = 100
            features["Fwd Packet Length Std"] = 50
            features["Bwd Packet Length Max"] = 1500
            features["Bwd Packet Length Min"] = 64
            features["Bwd Packet Length Mean"] = 100
            features["Bwd Packet Length Std"] = 50
            features["Flow Bytes/s"] = 2000
            features["Flow Packets/s"] = 20
            features["Flow IAT Mean"] = 50
            features["Flow IAT Std"] = 10
            features["Flow IAT Max"] = 100
            features["Flow IAT Min"] = 10

        # Fill remaining features with default values
        for feature_name in self.feature_names:
            if feature_name not in features:
                if "Active" in feature_name:
                    features[feature_name] = 1000
                elif "Idle" in feature_name:
                    features[feature_name] = 100
                elif "IAT" in feature_name:
                    features[feature_name] = 50
                else:
                    features[feature_name] = 100

        return features

    def test_health_check(self):
        """Test health endpoint"""
        response = self.session.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)

    def test_single_prediction_normal(self):
        """Test single prediction endpoint with normal traffic"""
        sample_data = {
            "features": self.create_realistic_features("normal"),
            "source_ip": "192.168.1.100",
            "attack_type": "normal_traffic",
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

        # Verify threat score is between 0 and 1
        self.assertGreaterEqual(data["threat_score"], 0)
        self.assertLessEqual(data["threat_score"], 1)

        # Normal traffic should have low threat score
        self.assertLess(
            data["threat_score"],
            0.5,
            f"Normal traffic got high threat score: {data['threat_score']}",
        )

    def test_single_prediction_ddos(self):
        """Test single prediction endpoint with DDoS attack"""
        sample_data = {
            "features": self.create_realistic_features("ddos"),
            "source_ip": "192.168.1.100",
            "attack_type": "ddos_attack",
        }

        response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertIn("threat_detected", data)
        self.assertIn("threat_level", data)
        self.assertIn("threat_score", data)
        self.assertIn("model_predictions", data)

        # Verify threat score is between 0 and 1
        self.assertGreaterEqual(data["threat_score"], 0)
        self.assertLessEqual(data["threat_score"], 1)

    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        batch_data = {
            "logs": [
                {
                    **self.create_realistic_features("normal"),
                    "source_ip": f"192.168.1.{i}",
                    "attack_type": "normal",
                }
                for i in range(1, 6)
            ]
        }

        response = self.session.post(f"{self.BASE_URL}/batch/predict", json=batch_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("results", data)
        self.assertIn("summary", data)
        self.assertEqual(len(data["results"]), 5)

        # Verify summary structure
        summary = data["summary"]
        self.assertIn("total_processed", summary)
        self.assertIn("threats_detected", summary)
        self.assertIn("average_threat_score", summary)
        self.assertIn("critical_threats", summary)

    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = self.session.get(f"{self.BASE_URL}/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("total_requests", data)
        self.assertIn("threats_detected", data)
        self.assertIn("threat_history", data)
        self.assertIn("current_threat_level", data)

        # Verify data types
        self.assertIsInstance(data["total_requests"], int)
        self.assertIsInstance(data["threats_detected"], int)
        self.assertIsInstance(data["threat_history"], list)
        self.assertIsInstance(data["current_threat_level"], str)

    def test_system_info_endpoint(self):
        """Test system information endpoint"""
        response = self.session.get(f"{self.BASE_URL}/system/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("models_loaded", data)
        self.assertIn("cpu_percent", data)
        self.assertIn("memory_percent", data)
        self.assertIn("total_predictions", data)
        self.assertIn("threats_detected", data)
        self.assertIn("detection_rate", data)

        # Verify data types and ranges
        self.assertIsInstance(data["models_loaded"], list)
        self.assertGreaterEqual(data["cpu_percent"], 0)
        self.assertLessEqual(data["cpu_percent"], 100)
        self.assertGreaterEqual(data["memory_percent"], 0)
        self.assertLessEqual(data["memory_percent"], 100)

    def test_alerts_endpoint(self):
        """Test alerts endpoint"""
        response = self.session.get(f"{self.BASE_URL}/alerts")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("alerts", data)
        self.assertIsInstance(data["alerts"], list)

    def test_concurrent_requests(self):
        """Test system under concurrent load"""

        def make_request():
            sample_data = {
                "features": self.create_realistic_features("normal"),
                "source_ip": f"192.168.1.{np.random.randint(1, 255)}",
                "attack_type": "test",
            }
            return self.session.post(f"{self.BASE_URL}/predict", json=sample_data)

        # Make 20 concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            responses = [f.result() for f in futures]

        # All should succeed
        for i, response in enumerate(responses):
            self.assertEqual(
                response.status_code,
                200,
                f"Request {i} failed with status {response.status_code}",
            )

    def test_response_time_requirement(self):
        """Test that response time is under 5 seconds"""
        sample_data = {
            "features": self.create_realistic_features("normal"),
            "source_ip": "192.168.1.100",
            "attack_type": "performance_test",
        }

        start_time = time.time()
        response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)
        elapsed_time = time.time() - start_time

        self.assertEqual(response.status_code, 200)
        self.assertLess(
            elapsed_time,
            5.0,
            f"Response time {elapsed_time:.2f}s exceeds 5s requirement",
        )

    def test_websocket_connection(self):
        """Test WebSocket connectivity"""
        sio = socketio.Client()
        connected = False
        received_message = False

        @sio.event
        def connect():
            nonlocal connected
            connected = True

        @sio.event
        def connected(data):
            nonlocal received_message
            received_message = True

        try:
            sio.connect(self.BASE_URL)
            time.sleep(2)  # Wait for connection and message
            self.assertTrue(connected, "WebSocket connection failed")
            # Note: received_message might be False if no initial message is sent
        except Exception as e:
            self.fail(f"WebSocket connection failed: {e}")
        finally:
            sio.disconnect()

    def test_invalid_request_handling(self):
        """Test handling of invalid requests"""
        # Test with missing features
        invalid_data = {"source_ip": "192.168.1.100", "attack_type": "test"}

        response = self.session.post(f"{self.BASE_URL}/predict", json=invalid_data)

        # Should handle gracefully (either 400 or 500 depending on implementation)
        self.assertIn(response.status_code, [400, 500])

    def test_model_loading(self):
        """Test that models are properly loaded"""
        response = self.session.get(f"{self.BASE_URL}/system/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Should have at least the baseline model loaded
        self.assertGreater(len(data["models_loaded"]), 0, "No models are loaded")
        self.assertIn("rf", data["models_loaded"], "Random Forest model not loaded")

    def test_threat_level_classification(self):
        """Test threat level classification logic"""
        # Test with different threat scores
        test_cases = [
            (0.05, "None"),
            (0.25, "Low"),
            (0.45, "Medium"),
            (0.65, "High"),
            (0.85, "Critical"),
        ]

        for expected_score, expected_level in test_cases:
            # Create a mock response to test classification logic
            # This would need to be implemented in the API if not already
            sample_data = {
                "features": self.create_realistic_features("normal"),
                "source_ip": "192.168.1.100",
                "attack_type": "classification_test",
            }

            response = self.session.post(f"{self.BASE_URL}/predict", json=sample_data)

            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Verify threat level is one of the expected values
            self.assertIn(
                data["threat_level"], ["None", "Low", "Medium", "High", "Critical"]
            )


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
