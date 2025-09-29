#!/usr/bin/env python3
"""
Unit tests for AI Cybersecurity Tool core components
Tests individual modules and functions in isolation
"""

import unittest
import numpy as np
import pandas as pd
import joblib
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.data_loader import CICIDSDataLoader
from src.preprocessor import DataPreprocessor
from src.models.xgboost_model import XGBoostDetector
from src.models.anomaly_detector import AnomalyDetector
from src.models.ensemble_model import EnsembleDetector
from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.performance_monitor import PerformanceMonitor


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality"""

    def setUp(self):
        self.loader = CICIDSDataLoader()

    def test_data_loader_initialization(self):
        """Test data loader initialization"""
        self.assertIsNotNone(self.loader.data_dir)
        self.assertTrue(self.loader.data_dir.exists())

    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        df = self.loader._generate_synthetic_data(1000)

        self.assertEqual(len(df), 1000)
        self.assertIn("Label", df.columns)
        self.assertGreater(len(df.columns), 70)  # Should have many features

        # Check label distribution
        label_counts = df["Label"].value_counts()
        self.assertIn("BENIGN", label_counts.index)
        self.assertIn("DDoS", label_counts.index)
        self.assertIn("PortScan", label_counts.index)

    @patch("pandas.read_csv")
    def test_load_friday_data_with_missing_files(self, mock_read_csv):
        """Test data loading when files are missing"""
        mock_read_csv.side_effect = FileNotFoundError()

        df = self.loader.load_friday_data(sample_size=100)

        self.assertEqual(len(df), 100)
        self.assertIn("Label", df.columns)


class TestDataPreprocessor(unittest.TestCase):
    """Test data preprocessing functionality"""

    def setUp(self):
        self.preprocessor = DataPreprocessor()
        # Create sample data
        self.sample_data = pd.DataFrame(
            {
                "Feature_1": np.random.randn(100),
                "Feature_2": np.random.randn(100),
                "Feature_3": np.random.randn(100),
                "Label": np.random.choice(["BENIGN", "DDoS"], 100),
            }
        )

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        self.assertIsNotNone(self.preprocessor)

    def test_process_data(self):
        """Test data processing pipeline"""
        X_train, X_test, y_train, y_test = self.preprocessor.process_data(
            self.sample_data
        )

        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), len(self.sample_data))
        self.assertEqual(len(y_train) + len(y_test), len(self.sample_data))

        # Check that features are numeric
        self.assertTrue(np.issubdtype(X_train.dtype, np.number))
        self.assertTrue(np.issubdtype(X_test.dtype, np.number))

        # Check that labels are binary
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        self.assertEqual(len(unique_labels), 2)


class TestXGBoostDetector(unittest.TestCase):
    """Test XGBoost detector functionality"""

    def setUp(self):
        self.detector = XGBoostDetector()
        # Create sample data
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.randint(0, 2, 20)

    def test_xgboost_initialization(self):
        """Test XGBoost detector initialization"""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.model.n_estimators, 200)
        self.assertEqual(self.detector.model.max_depth, 6)

    def test_xgboost_training(self):
        """Test XGBoost training"""
        model = self.detector.train(self.X_train, self.y_train)

        self.assertIsNotNone(model)
        self.assertIsNotNone(self.detector.feature_importance)
        self.assertEqual(len(self.detector.feature_importance), 10)

    def test_xgboost_prediction(self):
        """Test XGBoost prediction"""
        self.detector.train(self.X_train, self.y_train)
        probabilities = self.detector.predict_proba(self.X_test)

        self.assertEqual(len(probabilities), len(self.X_test))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

    def test_threshold_optimization(self):
        """Test threshold optimization"""
        self.detector.train(self.X_train, self.y_train)
        threshold = self.detector.optimize_threshold(self.X_test, self.y_test)

        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 1)


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detector functionality"""

    def setUp(self):
        self.detector = AnomalyDetector(contamination=0.1)
        # Create sample data
        self.X_normal = np.random.randn(100, 10)
        self.X_test = np.random.randn(20, 10)

    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization"""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.model.contamination, 0.1)
        self.assertIsNotNone(self.detector.scaler)

    def test_anomaly_detector_training(self):
        """Test anomaly detector training"""
        detector = self.detector.train(self.X_normal)

        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.model)

    def test_anomaly_detection(self):
        """Test anomaly detection"""
        self.detector.train(self.X_normal)
        predictions, scores = self.detector.detect_anomalies(self.X_test)

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(len(scores), len(self.X_test))

        # Predictions should be binary (0 or 1)
        unique_predictions = np.unique(predictions)
        self.assertTrue(np.all(np.isin(unique_predictions, [0, 1])))

    def test_anomaly_percentage(self):
        """Test anomaly percentage calculation"""
        self.detector.train(self.X_normal)
        percentage = self.detector.get_anomaly_percentage(self.X_test)

        self.assertGreaterEqual(percentage, 0)
        self.assertLessEqual(percentage, 100)


class TestEnsembleDetector(unittest.TestCase):
    """Test ensemble detector functionality"""

    def setUp(self):
        # Create mock models
        self.mock_model1 = MagicMock()
        self.mock_model1.predict_proba.return_value = np.array([[0.3, 0.7]])

        self.mock_model2 = MagicMock()
        self.mock_model2.detect_anomalies.return_value = (
            np.array([1]),
            np.array([-0.5]),
        )

        self.models = {
            "xgboost": self.mock_model1,
            "isolation_forest": self.mock_model2,
        }

        self.ensemble = EnsembleDetector(self.models)
        self.ensemble.set_weights({"xgboost": 0.6, "isolation_forest": 0.4})

        self.X_test = np.random.randn(1, 10)

    def test_ensemble_initialization(self):
        """Test ensemble detector initialization"""
        self.assertEqual(len(self.ensemble.models), 2)
        self.assertIn("xgboost", self.ensemble.models)
        self.assertIn("isolation_forest", self.ensemble.models)

    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        final_predictions, individual_predictions = self.ensemble.predict_ensemble(
            self.X_test
        )

        self.assertEqual(len(final_predictions), 1)
        self.assertIn("xgboost", individual_predictions)
        self.assertIn("isolation_forest", individual_predictions)

        # Final prediction should be weighted average
        self.assertGreaterEqual(final_predictions[0], 0)
        self.assertLessEqual(final_predictions[0], 1)

    def test_threat_level_classification(self):
        """Test threat level classification"""
        scores = [0.05, 0.25, 0.45, 0.65, 0.85]
        expected_levels = ["None", "Low", "Medium", "High", "Critical"]

        threat_levels = self.ensemble.classify_threat_level(scores)

        self.assertEqual(len(threat_levels), len(scores))
        for i, expected in enumerate(expected_levels):
            self.assertEqual(threat_levels[i], expected)


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluator functionality"""

    def setUp(self):
        self.evaluator = ModelEvaluator()
        # Create sample data
        self.y_true = np.random.randint(0, 2, 100)
        self.y_pred = np.random.randint(0, 2, 100)
        self.y_proba = np.random.rand(100)

    def test_evaluator_initialization(self):
        """Test model evaluator initialization"""
        self.assertIsNotNone(self.evaluator.results)
        self.assertEqual(len(self.evaluator.results), 0)

    def test_model_evaluation(self):
        """Test model evaluation"""
        results = self.evaluator.evaluate_model(
            self.y_true, self.y_pred, self.y_proba, "TestModel"
        )

        self.assertIn("model_name", results)
        self.assertIn("accuracy", results)
        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("f1_score", results)
        self.assertIn("roc_auc", results)
        self.assertIn("confusion_matrix", results)

        # Check that metrics are between 0 and 1
        self.assertGreaterEqual(results["accuracy"], 0)
        self.assertLessEqual(results["accuracy"], 1)
        self.assertGreaterEqual(results["precision"], 0)
        self.assertLessEqual(results["precision"], 1)

    def test_model_comparison(self):
        """Test model comparison"""
        # Add multiple models
        self.evaluator.evaluate_model(self.y_true, self.y_pred, self.y_proba, "Model1")
        self.evaluator.evaluate_model(self.y_true, self.y_pred, self.y_proba, "Model2")

        comparison_df = self.evaluator.compare_models()

        self.assertIsNotNone(comparison_df)
        self.assertEqual(len(comparison_df), 2)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitor functionality"""

    def setUp(self):
        self.monitor = PerformanceMonitor(window_size=10)
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([1, 0, 1])
        self.X_batch = np.random.randn(3, 10)

    def test_monitor_initialization(self):
        """Test performance monitor initialization"""
        self.assertEqual(self.monitor.window_size, 10)
        self.assertEqual(len(self.monitor.inference_times), 0)
        self.assertEqual(len(self.monitor.memory_usage), 0)
        self.assertEqual(len(self.monitor.cpu_usage), 0)

    def test_inference_time_measurement(self):
        """Test inference time measurement"""
        inference_time = self.monitor.measure_inference_time(
            self.mock_model, self.X_batch
        )

        self.assertGreater(inference_time, 0)
        self.assertEqual(len(self.monitor.inference_times), 1)

    def test_system_metrics(self):
        """Test system metrics collection"""
        metrics = self.monitor.get_system_metrics()

        self.assertIn("cpu_percent", metrics)
        self.assertIn("memory_percent", metrics)
        self.assertIn("memory_mb", metrics)

        self.assertGreaterEqual(metrics["cpu_percent"], 0)
        self.assertLessEqual(metrics["cpu_percent"], 100)
        self.assertGreaterEqual(metrics["memory_percent"], 0)
        self.assertLessEqual(metrics["memory_percent"], 100)

    def test_performance_summary(self):
        """Test performance summary"""
        # Add some data
        self.monitor.measure_inference_time(self.mock_model, self.X_batch)
        self.monitor.get_system_metrics()

        summary = self.monitor.get_performance_summary()

        if summary:  # Only test if we have data
            self.assertIn("avg_inference_time_ms", summary)
            self.assertIn("p95_inference_time_ms", summary)
            self.assertIn("p99_inference_time_ms", summary)
            self.assertIn("avg_cpu_percent", summary)
            self.assertIn("avg_memory_percent", summary)
            self.assertIn("throughput_per_second", summary)

    def test_performance_requirements(self):
        """Test performance requirements check"""
        # Add some data
        self.monitor.measure_inference_time(self.mock_model, self.X_batch)

        meets_requirement = self.monitor.check_performance_requirements(
            max_latency_ms=1000
        )

        self.assertIsInstance(meets_requirement, bool)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
