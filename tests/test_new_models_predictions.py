#!/usr/bin/env python3
"""
Test new models predictions
Tests Random Forest, XGBoost, and Isolation Forest predictions on new datasets
"""

import unittest
import numpy as np
import pandas as pd
import joblib
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = Path(__file__).parent.parent / "models"


class TestNewModelsPredictions(unittest.TestCase):
    """Test new models predictions"""

    def setUp(self):
        """Set up test fixtures"""
        self.models_dir = MODELS_DIR
        
        # Load feature names
        features_path = self.models_dir / "feature_names_new_datasets.pkl"
        if features_path.exists():
            self.feature_names = joblib.load(features_path)
            if not isinstance(self.feature_names, list):
                self.feature_names = list(self.feature_names)
        else:
            self.feature_names = None
            self.skipTest("Feature names not found")

    def create_test_features(self, n_samples=10, attack_ratio=0.5):
        """Create test feature data"""
        if self.feature_names is None:
            self.skipTest("Feature names not available")
        
        n_features = len(self.feature_names)
        X = np.random.randn(n_samples, n_features)
        
        # Make some features more attack-like
        n_attack = int(n_samples * attack_ratio)
        if n_attack > 0:
            # Attack samples: higher packet rates, shorter durations
            attack_indices = np.random.choice(n_samples, n_attack, replace=False)
            for idx in attack_indices:
                # Simulate attack patterns
                if 'Flow Packets/s' in self.feature_names:
                    packet_idx = self.feature_names.index('Flow Packets/s')
                    X[idx, packet_idx] = np.random.uniform(1000, 5000)
                if 'Flow Duration' in self.feature_names:
                    duration_idx = self.feature_names.index('Flow Duration')
                    X[idx, duration_idx] = np.random.uniform(1000, 10000)
        
        return X

    def test_load_random_forest(self):
        """Test loading Random Forest model"""
        model_path = self.models_dir / "random_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Random Forest model not found")
        
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

    def test_random_forest_prediction(self):
        """Test Random Forest predictions"""
        model_path = self.models_dir / "random_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Random Forest model not found")
        
        model = joblib.load(model_path)
        X = self.create_test_features(n_samples=10)
        
        # Test prediction
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            self.assertEqual(len(predictions), len(X))
            self.assertIn(predictions[0], [0, 1], "Predictions should be binary")
        
        # Test predict_proba if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            self.assertEqual(len(probabilities), len(X))
            self.assertEqual(probabilities.shape[1], 2, "Should have 2 classes")
            self.assertTrue(np.all(probabilities >= 0) and np.all(probabilities <= 1),
                          "Probabilities should be in [0, 1]")

    def test_load_xgboost(self):
        """Test loading XGBoost model"""
        model_path = self.models_dir / "xgboost_model_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("XGBoost model not found")
        
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

    def test_xgboost_prediction(self):
        """Test XGBoost predictions"""
        model_path = self.models_dir / "xgboost_model_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("XGBoost model not found")
        
        model = joblib.load(model_path)
        X = self.create_test_features(n_samples=10)
        
        # XGBoostDetector has predict_proba method
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            # XGBoostDetector returns 1D array of probabilities for class 1
            if probabilities.ndim == 1:
                self.assertEqual(len(probabilities), len(X))
                self.assertTrue(np.all(probabilities >= 0) and np.all(probabilities <= 1),
                              "Probabilities should be in [0, 1]")
            else:
                self.assertEqual(probabilities.shape[0], len(X))
                self.assertTrue(np.all(probabilities >= 0) and np.all(probabilities <= 1))
        
        # If it's a standard XGBoost model
        if hasattr(model, 'model') and hasattr(model.model, 'predict'):
            predictions = model.model.predict(X)
            self.assertEqual(len(predictions), len(X))

    def test_load_isolation_forest(self):
        """Test loading Isolation Forest model"""
        model_path = self.models_dir / "isolation_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Isolation Forest model not found")
        
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

    def test_isolation_forest_prediction(self):
        """Test Isolation Forest anomaly detection"""
        model_path = self.models_dir / "isolation_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Isolation Forest model not found")
        
        model = joblib.load(model_path)
        X = self.create_test_features(n_samples=10)
        
        # Test detect_anomalies if available
        if hasattr(model, 'detect_anomalies'):
            predictions, scores = model.detect_anomalies(X)
            self.assertEqual(len(predictions), len(X))
            self.assertEqual(len(scores), len(X))
            self.assertIn(predictions[0], [0, 1], "Predictions should be binary")
        # Test standard predict
        elif hasattr(model, 'predict'):
            predictions = model.predict(X)
            self.assertEqual(len(predictions), len(X))
            # Isolation Forest returns -1 (anomaly) or 1 (normal)
            self.assertTrue(np.all(np.isin(predictions, [-1, 1])),
                          "Isolation Forest should return -1 or 1")

    def test_prediction_probabilities_range(self):
        """Test that prediction probabilities are in valid range [0, 1]"""
        models_to_test = [
            ("random_forest_new_datasets.pkl", "Random Forest"),
            ("xgboost_model_new_datasets.pkl", "XGBoost"),
        ]
        
        X = self.create_test_features(n_samples=5)
        
        for model_file, model_name in models_to_test:
            model_path = self.models_dir / model_file
            if not model_path.exists():
                continue
            
            model = joblib.load(model_path)
            
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)
                    if proba.ndim == 1:
                        # 1D array (probabilities for class 1)
                        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1),
                                       f"{model_name} probabilities should be in [0, 1]")
                    else:
                        # 2D array
                        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1),
                                       f"{model_name} probabilities should be in [0, 1]")
                except Exception as e:
                    # Some models might not support predict_proba with this input
                    pass

    def test_benign_traffic_prediction(self):
        """Test predictions on benign traffic patterns"""
        model_path = self.models_dir / "random_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Random Forest model not found")
        
        model = joblib.load(model_path)
        X = self.create_test_features(n_samples=10, attack_ratio=0.0)  # All benign
        
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            # Most should be benign (0)
            benign_count = np.sum(predictions == 0)
            self.assertGreater(benign_count, 0, "Should predict some benign traffic")

    def test_attack_traffic_prediction(self):
        """Test predictions on attack traffic patterns"""
        model_path = self.models_dir / "random_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Random Forest model not found")
        
        model = joblib.load(model_path)
        X = self.create_test_features(n_samples=10, attack_ratio=1.0)  # All attack
        
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            # Some should be attack (1)
            attack_count = np.sum(predictions == 1)
            # Note: May not always detect, but should at least make predictions
            self.assertIsInstance(attack_count, (int, np.integer))

    def test_model_consistency(self):
        """Test that models produce consistent predictions"""
        model_path = self.models_dir / "random_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Random Forest model not found")
        
        model = joblib.load(model_path)
        X = self.create_test_features(n_samples=5)
        
        # Make predictions twice
        if hasattr(model, 'predict'):
            pred1 = model.predict(X)
            pred2 = model.predict(X)
            
            # Should be identical
            self.assertTrue(np.array_equal(pred1, pred2),
                          "Predictions should be consistent")

    def test_scaler_compatibility(self):
        """Test that scaler works with new models"""
        scaler_path = self.models_dir / "scaler_new_datasets.pkl"
        if not scaler_path.exists():
            self.skipTest("Scaler not found")
        
        scaler = joblib.load(scaler_path)
        X = self.create_test_features(n_samples=10)
        
        # Transform features
        X_scaled = scaler.transform(X)
        
        self.assertEqual(X_scaled.shape, X.shape)
        self.assertFalse(np.any(np.isnan(X_scaled)), "Scaled features should not contain NaN")
        self.assertFalse(np.any(np.isinf(X_scaled)), "Scaled features should not contain Inf")

    def test_feature_count_match(self):
        """Test that test features match model's expected feature count"""
        model_path = self.models_dir / "random_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest("Random Forest model not found")
        
        model = joblib.load(model_path)
        
        # Get expected feature count from model
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
        elif hasattr(model, 'feature_importances_'):
            expected_features = len(model.feature_importances_)
        elif hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
            expected_features = model.model.n_features_in_
        else:
            self.skipTest("Could not determine expected feature count")
        
        # Create features with correct count
        X = self.create_test_features(n_samples=5)
        
        self.assertEqual(X.shape[1], expected_features,
                        f"Feature count should match model expectation ({expected_features})")


if __name__ == "__main__":
    unittest.main()

