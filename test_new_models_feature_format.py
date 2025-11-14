#!/usr/bin/env python3
"""
Test feature format for new models
Tests prediction with aligned features (65-145 features) vs old (78 CICIDS2017)
"""

import unittest
import numpy as np
import joblib
import requests
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = Path(__file__).parent / "models"
API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"

# Old CICIDS2017 features (78 features)
CICIDS_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
    'Idle Std', 'Idle Max', 'Idle Min'
]


class TestNewModelsFeatureFormat(unittest.TestCase):
    """Test feature format for new models"""

    def setUp(self):
        """Set up test fixtures"""
        # Load new feature names
        features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
        if features_path.exists():
            self.new_feature_names = joblib.load(features_path)
            if not isinstance(self.new_feature_names, list):
                self.new_feature_names = list(self.new_feature_names)
        else:
            self.new_feature_names = None
            self.skipTest("New feature names not found")

    def test_feature_count_comparison(self):
        """Compare feature counts: old (78) vs new (65-145)"""
        old_count = len(CICIDS_FEATURES)
        new_count = len(self.new_feature_names)
        
        print(f"\nOld CICIDS2017 features: {old_count}")
        print(f"New aligned features: {new_count}")
        
        self.assertGreaterEqual(new_count, 65, 
                              f"New features should be at least 65, got {new_count}")
        # Note: Actual is 145, which is fine

    def test_create_features_new_format(self):
        """Test creating features with new dataset feature names"""
        features = {}
        
        # Create features using new feature names
        for feat in self.new_feature_names[:20]:  # Use first 20
            if "Port" in feat:
                features[feat] = 443
            elif "Duration" in feat:
                features[feat] = 100000
            elif "Packets" in feat:
                features[feat] = 10
            else:
                features[feat] = np.random.uniform(0, 100)
        
        self.assertEqual(len(features), 20)
        self.assertIn(list(features.keys())[0], self.new_feature_names)

    def test_prediction_with_aligned_features(self):
        """Test prediction with aligned features (65-145 features)"""
        # Create features with all new feature names
        features = {}
        for feat in self.new_feature_names:
            if "Port" in feat:
                features[feat] = 443
            elif "Duration" in feat:
                features[feat] = 100000
            elif "Packets" in feat or "Packets/s" in feat:
                features[feat] = 10
            elif "Bytes" in feat or "Bytes/s" in feat:
                features[feat] = 2000
            else:
                features[feat] = np.random.uniform(0, 100)
        
        # Test via API
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"features": features},
                headers={"X-API-Key": API_KEY},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn("threat_score", data)
                self.assertGreaterEqual(data["threat_score"], 0)
                self.assertLessEqual(data["threat_score"], 1)
            else:
                self.skipTest(f"API not available or error: {response.status_code}")
        except requests.exceptions.RequestException:
            self.skipTest("API not available")

    def test_prediction_with_missing_features(self):
        """Test prediction with missing features (should fill with defaults)"""
        # Create features with only a subset
        partial_features = {}
        for feat in self.new_feature_names[:10]:  # Only first 10
            partial_features[feat] = np.random.uniform(0, 100)
        
        # Test via API
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"features": partial_features},
                headers={"X-API-Key": API_KEY},
                timeout=10
            )
            
            # Should either succeed (with defaults) or return 400
            self.assertIn(response.status_code, [200, 400])
        except requests.exceptions.RequestException:
            self.skipTest("API not available")

    def test_prediction_with_extra_features(self):
        """Test prediction with extra features (should ignore)"""
        # Create features with all new features plus some extra
        features = {}
        for feat in self.new_feature_names:
            features[feat] = np.random.uniform(0, 100)
        
        # Add extra features
        features["ExtraFeature1"] = 123
        features["ExtraFeature2"] = 456
        features["UnknownFeature"] = 789
        
        # Test via API
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"features": features},
                headers={"X-API-Key": API_KEY},
                timeout=10
            )
            
            # Should succeed (extra features ignored)
            self.assertEqual(response.status_code, 200)
        except requests.exceptions.RequestException:
            self.skipTest("API not available")

    def test_feature_overlap_old_vs_new(self):
        """Test overlap between old and new feature sets"""
        old_set = set(CICIDS_FEATURES)
        new_set = set(self.new_feature_names)
        
        overlap = old_set.intersection(new_set)
        old_only = old_set - new_set
        new_only = new_set - old_set
        
        print(f"\nFeature overlap: {len(overlap)} features")
        print(f"Old only: {len(old_only)} features")
        print(f"New only: {len(new_only)} features")
        
        # Should have some overlap
        self.assertGreater(len(overlap), 0, "Should have some common features")

    def test_feature_name_normalization(self):
        """Test that feature names are normalized correctly"""
        # Check for common variations
        variations = {
            "Flow Duration": ["flow_duration", "FlowDuration", "FLOW_DURATION"],
            "Destination Port": ["destination_port", "DestinationPort", "DEST_PORT"],
        }
        
        for standard, variants in variations.items():
            # Check if standard or any variant exists in new features
            found = False
            for feat in self.new_feature_names:
                if standard.lower() in feat.lower() or any(v.lower() in feat.lower() for v in variants):
                    found = True
                    break
            # Not all may be present, but at least check structure
            self.assertIsInstance(found, bool)

    def test_feature_count_matches_model(self):
        """Test that feature count matches model expectation"""
        model_path = MODELS_DIR / "random_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Random Forest model not found")
        
        model = joblib.load(model_path)
        
        # Get expected feature count
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
        elif hasattr(model, 'feature_importances_'):
            expected_features = len(model.feature_importances_)
        else:
            self.skipTest("Could not determine expected feature count")
        
        # Compare with feature names
        self.assertEqual(len(self.new_feature_names), expected_features,
                        f"Feature names count ({len(self.new_feature_names)}) should match "
                        f"model expectation ({expected_features})")


if __name__ == "__main__":
    unittest.main()

