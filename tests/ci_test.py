#!/usr/bin/env python3
"""
CI-specific tests that don't require trained models
These tests run in GitHub Actions CI environment
"""

import unittest
import sys
import os
import warnings

# Suppress warnings for cleaner CI output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestCIImports(unittest.TestCase):
    """Test that all modules can be imported without errors"""
    
    def test_import_data_loader(self):
        """Test data loader import"""
        try:
            from src.data_loader import CICIDSDataLoader
            loader = CICIDSDataLoader()
            self.assertIsNotNone(loader)
        except Exception as e:
            self.fail(f"Failed to import CICIDSDataLoader: {e}")
    
    def test_import_preprocessor(self):
        """Test preprocessor import"""
        try:
            from src.preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
            self.assertIsNotNone(preprocessor)
        except Exception as e:
            self.fail(f"Failed to import DataPreprocessor: {e}")
    
    def test_import_ssl_modules(self):
        """Test SSL modules import"""
        try:
            from src.models.ssl_enhancement import SSLEnhancement
            from src.models.integrate_ssl import SSLIntegratedModel
            from src.models.ssl_api_integration import SSLAPIIntegration
            self.assertTrue(True)  # If we get here, imports worked
        except Exception as e:
            self.fail(f"Failed to import SSL modules: {e}")
    
    def test_import_evaluation_modules(self):
        """Test evaluation modules import"""
        try:
            from src.evaluation.model_evaluator import ModelEvaluator
            from src.evaluation.performance_monitor import PerformanceMonitor
            self.assertTrue(True)  # If we get here, imports worked
        except Exception as e:
            self.fail(f"Failed to import evaluation modules: {e}")


class TestCIBasicFunctionality(unittest.TestCase):
    """Test basic functionality that doesn't require trained models"""
    
    def test_data_loader_basic(self):
        """Test basic data loader functionality"""
        from src.data_loader import CICIDSDataLoader
        loader = CICIDSDataLoader()
        
        # Test that we can create the loader
        self.assertIsNotNone(loader)
        self.assertIsNotNone(loader.data_dir)
    
    def test_preprocessor_basic(self):
        """Test basic preprocessor functionality"""
        from src.preprocessor import DataPreprocessor
        import pandas as pd
        import numpy as np
        
        preprocessor = DataPreprocessor()
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Label': ['BENIGN', 'BENIGN', 'BENIGN', 'BENIGN', 'BENIGN']
        })
        
        # Test cleaning
        cleaned_data = preprocessor.clean_data(test_data.copy())
        self.assertEqual(len(cleaned_data), 5)
        
        # Test feature preparation
        X, y = preprocessor.prepare_features(test_data.copy())
        self.assertEqual(X.shape[0], 5)
        self.assertEqual(len(y), 5)
    
    def test_ssl_enhancement_basic(self):
        """Test basic SSL enhancement functionality"""
        from src.models.ssl_enhancement import SSLEnhancement
        import numpy as np
        
        # Create test data
        X = np.random.randn(100, 10)
        
        # Test SSL initialization
        ssl = SSLEnhancement(input_dim=10, output_dim=5)
        self.assertIsNotNone(ssl)
        self.assertEqual(ssl.input_dim, 10)
        self.assertEqual(ssl.output_dim, 5)


if __name__ == '__main__':
    # Run CI tests
    unittest.main(verbosity=2)
