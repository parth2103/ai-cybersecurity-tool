#!/usr/bin/env python3
"""
Quick SSL Performance Test

A simplified version of the SSL performance comparison that avoids infinite loops
and focuses on essential performance metrics.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
import logging
import warnings

# Suppress sklearn warnings to prevent infinite loops
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Add src to path
sys.path.append('src')

from models.ssl_api_integration import ssl_api, predict_with_ssl, compare_ssl_baseline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_performance():
    """Test basic SSL vs baseline performance."""
    logger.info("Testing basic SSL vs baseline performance")
    
    test_features = {
        'Destination Port': 80,
        'Flow Duration': 1000000,
        'Total Fwd Packets': 10000,
        'Total Backward Packets': 10000,
        'source_ip': '192.168.1.100',
        'attack_type': 'DDoS_Test'
    }
    
    results = {
        'baseline_times': [],
        'ssl_times': [],
        'predictions_match': []
    }
    
    # Test with multiple iterations
    num_tests = 10
    logger.info(f"Running {num_tests} prediction tests...")
    
    for i in range(num_tests):
        # Test baseline
        start = time.time()
        baseline_result = predict_with_ssl(test_features, use_ssl=False)
        baseline_time = time.time() - start
        
        # Test SSL
        start = time.time()
        ssl_result = predict_with_ssl(test_features, use_ssl=True)
        ssl_time = time.time() - start
        
        results['baseline_times'].append(baseline_time)
        results['ssl_times'].append(ssl_time)
        results['predictions_match'].append(
            baseline_result['prediction'] == ssl_result['prediction']
        )
        
        if (i + 1) % 5 == 0:
            logger.info(f"Completed {i + 1}/{num_tests} tests")
    
    return results


def test_model_loading():
    """Test model loading performance."""
    logger.info("Testing model loading performance")
    
    start = time.time()
    status = ssl_api.get_model_info()
    loading_time = time.time() - start
    
    return {
        'loading_time': loading_time,
        'model_status': status
    }


def test_feature_comparison():
    """Test feature comparison functionality."""
    logger.info("Testing feature comparison")
    
    test_features = {
        'Destination Port': 80,
        'Flow Duration': 1000000,
        'Total Fwd Packets': 10000,
        'Total Backward Packets': 10000
    }
    
    start = time.time()
    comparison = compare_ssl_baseline(test_features)
    comparison_time = time.time() - start
    
    return {
        'comparison_time': comparison_time,
        'comparison_result': comparison
    }


def run_quick_performance_test():
    """Run quick performance test."""
    logger.info("Starting quick SSL performance test")
    
    results = {
        'model_loading': test_model_loading(),
        'basic_performance': test_basic_performance(),
        'feature_comparison': test_feature_comparison()
    }
    
    # Calculate summary statistics
    baseline_times = results['basic_performance']['baseline_times']
    ssl_times = results['basic_performance']['ssl_times']
    predictions_match = results['basic_performance']['predictions_match']
    
    results['summary'] = {
        'average_baseline_time': np.mean(baseline_times),
        'average_ssl_time': np.mean(ssl_times),
        'time_overhead': np.mean(ssl_times) - np.mean(baseline_times),
        'relative_overhead': ((np.mean(ssl_times) - np.mean(baseline_times)) / np.mean(baseline_times)) * 100,
        'prediction_match_rate': np.mean(predictions_match),
        'total_tests': len(baseline_times)
    }
    
    return results


def print_results(results):
    """Print performance test results."""
    print("\n" + "="*60)
    print("QUICK SSL PERFORMANCE TEST RESULTS")
    print("="*60)
    
    # Model status
    model_status = results['model_loading']['model_status']
    print(f"Model Status:")
    print(f"  Baseline Model: {'✓' if model_status['baseline_model_loaded'] else '✗'}")
    print(f"  SSL Encoder: {'✓' if model_status['ssl_encoder_loaded'] else '✗'}")
    print(f"  Enhanced Model: {'✓' if model_status['enhanced_model_loaded'] else '✗'}")
    print(f"  SSL Available: {'✓' if model_status['ssl_available'] else '✗'}")
    
    # Performance summary
    summary = results['summary']
    print(f"\nPerformance Summary:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Average Baseline Time: {summary['average_baseline_time']:.4f}s")
    print(f"  Average SSL Time: {summary['average_ssl_time']:.4f}s")
    print(f"  Time Overhead: {summary['time_overhead']:.4f}s ({summary['relative_overhead']:.1f}%)")
    print(f"  Prediction Match Rate: {summary['prediction_match_rate']:.1%}")
    
    # Model loading time
    loading_time = results['model_loading']['loading_time']
    print(f"\nModel Loading Time: {loading_time:.4f}s")
    
    # Feature comparison
    comparison_time = results['feature_comparison']['comparison_time']
    print(f"Feature Comparison Time: {comparison_time:.4f}s")
    
    print("="*60)


def main():
    """Main function for quick SSL performance test."""
    logger.info("Starting quick SSL performance test")
    
    try:
        results = run_quick_performance_test()
        
        # Print results
        print_results(results)
        
        # Save results
        results_path = 'results/quick_ssl_performance_results.json'
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        logger.info("Quick SSL performance test completed successfully")
        
    except Exception as e:
        logger.error(f"Quick performance test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
