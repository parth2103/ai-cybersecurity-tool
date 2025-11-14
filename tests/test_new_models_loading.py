#!/usr/bin/env python3
"""
Test new models loading verification
Verifies that new models (IoT-IDAD 2024 + CICAPT-IIOT) load correctly
"""

import unittest
import joblib
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = Path(__file__).parent.parent / "models"


class TestNewModelsLoading(unittest.TestCase):
    """Test new models loading and verification"""

    def test_new_random_forest_exists(self):
        """Test that new Random Forest model exists"""
        model_path = MODELS_DIR / "random_forest_new_datasets.pkl"
        self.assertTrue(
            model_path.exists(),
            f"New Random Forest model not found at {model_path}"
        )

    def test_new_xgboost_exists(self):
        """Test that new XGBoost model exists"""
        model_path = MODELS_DIR / "xgboost_model_new_datasets.pkl"
        self.assertTrue(
            model_path.exists(),
            f"New XGBoost model not found at {model_path}"
        )

    def test_new_isolation_forest_exists(self):
        """Test that new Isolation Forest model exists"""
        model_path = MODELS_DIR / "isolation_forest_new_datasets.pkl"
        self.assertTrue(
            model_path.exists(),
            f"New Isolation Forest model not found at {model_path}"
        )

    def test_new_scaler_exists(self):
        """Test that new scaler exists"""
        scaler_path = MODELS_DIR / "scaler_new_datasets.pkl"
        self.assertTrue(
            scaler_path.exists(),
            f"New scaler not found at {scaler_path}"
        )

    def test_new_feature_names_exists(self):
        """Test that new feature names file exists"""
        features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
        self.assertTrue(
            features_path.exists(),
            f"New feature names not found at {features_path}"
        )

    def test_load_new_random_forest(self):
        """Test loading new Random Forest model"""
        model_path = MODELS_DIR / "random_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest(f"Model not found: {model_path}")
        
        try:
            model = joblib.load(model_path)
            self.assertIsNotNone(model, "Failed to load Random Forest model")
            
            # Verify it's a RandomForestClassifier or similar
            model_type = type(model).__name__
            self.assertIn("RandomForest", model_type, 
                         f"Expected RandomForest model, got {model_type}")
        except Exception as e:
            self.fail(f"Error loading Random Forest model: {e}")

    def test_load_new_xgboost(self):
        """Test loading new XGBoost model"""
        model_path = MODELS_DIR / "xgboost_model_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest(f"Model not found: {model_path}")
        
        try:
            model = joblib.load(model_path)
            self.assertIsNotNone(model, "Failed to load XGBoost model")
            
            # Verify it's XGBoostDetector or XGBClassifier
            model_type = type(model).__name__
            self.assertIn("XGB", model_type, 
                         f"Expected XGBoost model, got {model_type}")
        except Exception as e:
            self.fail(f"Error loading XGBoost model: {e}")

    def test_load_new_isolation_forest(self):
        """Test loading new Isolation Forest model"""
        model_path = MODELS_DIR / "isolation_forest_new_datasets.pkl"
        if not model_path.exists():
            self.skipTest(f"Model not found: {model_path}")
        
        try:
            model = joblib.load(model_path)
            self.assertIsNotNone(model, "Failed to load Isolation Forest model")
            
            # Verify it's AnomalyDetector or IsolationForest
            model_type = type(model).__name__
            self.assertIn("Isolation" in model_type or "Anomaly" in model_type, 
                         f"Expected Isolation Forest model, got {model_type}")
        except Exception as e:
            self.fail(f"Error loading Isolation Forest model: {e}")

    def test_load_new_scaler(self):
        """Test loading new scaler"""
        scaler_path = MODELS_DIR / "scaler_new_datasets.pkl"
        if not scaler_path.exists():
            self.skipTest(f"Scaler not found: {scaler_path}")
        
        try:
            scaler = joblib.load(scaler_path)
            self.assertIsNotNone(scaler, "Failed to load scaler")
            
            # Verify it's a StandardScaler or similar
            scaler_type = type(scaler).__name__
            self.assertIn("Scaler", scaler_type, 
                         f"Expected Scaler, got {scaler_type}")
        except Exception as e:
            self.fail(f"Error loading scaler: {e}")

    def test_load_new_feature_names(self):
        """Test loading new feature names"""
        features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
        if not features_path.exists():
            self.skipTest(f"Feature names not found: {features_path}")
        
        try:
            feature_names = joblib.load(features_path)
            self.assertIsNotNone(feature_names, "Failed to load feature names")
            
            # Convert to list if needed
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            
            self.assertGreater(len(feature_names), 0, 
                             "Feature names list is empty")
            self.assertGreaterEqual(len(feature_names), 65, 
                                  f"Expected at least 65 features, got {len(feature_names)}")
            
            # All should be strings
            for feat in feature_names[:10]:  # Check first 10
                self.assertIsInstance(feat, str, 
                                    f"Feature name should be string, got {type(feat)}")
        except Exception as e:
            self.fail(f"Error loading feature names: {e}")

    def test_feature_count_matches_expectation(self):
        """Test that feature count is in expected range (65-145 features, aligned)"""
        features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
        if not features_path.exists():
            self.skipTest(f"Feature names not found: {features_path}")
        
        feature_names = joblib.load(features_path)
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)
        
        feature_count = len(feature_names)
        self.assertGreaterEqual(feature_count, 65, 
                              f"Expected at least 65 features, got {feature_count}")
        # Note: Actual count is 145, which is fine for aligned features

    def test_models_have_predict_method(self):
        """Test that loaded models have predict or predict_proba methods"""
        models_to_test = [
            ("random_forest_new_datasets.pkl", "Random Forest"),
            ("xgboost_model_new_datasets.pkl", "XGBoost"),
            ("isolation_forest_new_datasets.pkl", "Isolation Forest"),
        ]
        
        for model_file, model_name in models_to_test:
            model_path = MODELS_DIR / model_file
            if not model_path.exists():
                continue
            
            model = joblib.load(model_path)
            
            # Check for predict or predict_proba
            has_predict = hasattr(model, 'predict')
            has_predict_proba = hasattr(model, 'predict_proba')
            has_detect_anomalies = hasattr(model, 'detect_anomalies')
            
            self.assertTrue(
                has_predict or has_predict_proba or has_detect_anomalies,
                f"{model_name} model must have predict, predict_proba, or detect_anomalies method"
            )


if __name__ == "__main__":
    unittest.main()

