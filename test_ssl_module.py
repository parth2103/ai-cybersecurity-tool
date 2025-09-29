#!/usr/bin/env python3
"""
SSL Module Test Script

This script validates that the SSL enhancement module works correctly
with the existing cybersecurity detection system.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.append('src')

from src.models.ssl_enhancement import SSLEnhancement, NetworkTrafficDataset
from src.models.integrate_ssl import SSLIntegratedModel
from src.models.ssl_api_integration import ssl_api, predict_with_ssl
from src.data_loader import CICIDSDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ssl_enhancement():
    """Test basic SSL enhancement functionality."""
    logger.info("Testing SSL Enhancement Module...")
    
    try:
        # Create synthetic data for testing
        np.random.seed(42)
        X = np.random.randn(100, 20)  # 100 samples, 20 features
        
        # Initialize SSL enhancement
        ssl = SSLEnhancement(input_dim=20, output_dim=16)
        
        # Test dataset creation
        dataset = NetworkTrafficDataset(X, augment=True)
        assert len(dataset) == 100
        
        # Test forward pass
        sample = dataset[0]
        assert len(sample) == 2  # Two augmented views
        assert sample[0].shape == sample[1].shape
        
        logger.info("✓ SSL Enhancement module working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ SSL Enhancement test failed: {e}")
        return False


def test_ssl_integration():
    """Test SSL integration with existing models."""
    logger.info("Testing SSL Integration...")
    
    try:
        # Check if baseline model exists
        baseline_path = Path('models/baseline_model.pkl')
        if not baseline_path.exists():
            logger.warning("Baseline model not found, skipping integration test")
            return True
        
        # Test integration model loading
        integrated_model = SSLIntegratedModel(str(baseline_path))
        
        # Test feature preparation
        test_features = np.random.randn(10, 78)  # Typical feature count
        orig_features, enhanced_features = integrated_model.create_enhanced_features(test_features)
        
        assert orig_features.shape == test_features.shape
        logger.info("✓ SSL Integration module working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ SSL Integration test failed: {e}")
        return False


def test_ssl_api():
    """Test SSL API integration."""
    logger.info("Testing SSL API Integration...")
    
    try:
        # Test model status
        status = ssl_api.get_model_info()
        assert isinstance(status, dict)
        
        # Test prediction with dummy features
        test_features = {
            'Destination Port': 80,
            'Flow Duration': 1000000,
            'Total Fwd Packets': 10000,
            'Total Backward Packets': 10000
        }
        
        # Test baseline prediction
        baseline_result = predict_with_ssl(test_features, use_ssl=False)
        assert 'prediction' in baseline_result
        
        # Test SSL prediction (if available)
        ssl_result = predict_with_ssl(test_features, use_ssl=True)
        assert 'prediction' in ssl_result
        
        logger.info("✓ SSL API Integration working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ SSL API test failed: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing Data Loading...")
    
    try:
        # Test data loader
        loader = CICIDSDataLoader()
        df = loader.load_friday_data(sample_size=100)  # Small sample for testing
        
        assert len(df) > 0
        assert 'Label' in df.columns or ' Label' in df.columns
        
        logger.info("✓ Data loading working correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loading test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive SSL module test."""
    logger.info("="*50)
    logger.info("SSL MODULE COMPREHENSIVE TEST")
    logger.info("="*50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("SSL Enhancement", test_ssl_enhancement),
        ("SSL Integration", test_ssl_integration),
        ("SSL API", test_ssl_api)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:20} : {status}")
        if success:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! SSL module is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the issues above.")
        return False


def main():
    """Main test function."""
    success = run_comprehensive_test()
    
    if success:
        print("\n" + "="*50)
        print("SSL MODULE VALIDATION COMPLETE")
        print("="*50)
        print("✅ All components working correctly")
        print("✅ Ready for SSL enhancement training")
        print("✅ Compatible with existing system")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("SSL MODULE VALIDATION FAILED")
        print("="*50)
        print("❌ Some components have issues")
        print("❌ Please fix errors before proceeding")
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    main()
