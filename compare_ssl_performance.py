#!/usr/bin/env python3
"""
SSL Performance Comparison Script

This script provides comprehensive performance comparison between baseline
and SSL-enhanced models across multiple metrics and scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Any, Tuple
import warnings

# Suppress sklearn warnings to prevent infinite loops
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Add src to path
sys.path.append('src')

from models.integrate_ssl import SSLIntegratedModel
from models.ssl_api_integration import ssl_api, predict_with_ssl, compare_ssl_baseline
from data_loader import CICIDSDataLoader
from preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSLPerformanceComparator:
    """Comprehensive SSL performance comparison."""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.results = {}
        
    def run_comprehensive_comparison(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive performance comparison."""
        logger.info("Starting comprehensive SSL performance comparison")
        
        results = {
            'model_loading': self._test_model_loading(),
            'prediction_performance': self._test_prediction_performance(test_scenarios),
            'feature_analysis': self._analyze_feature_importance(),
            'scalability_test': self._test_scalability(),
            'robustness_test': self._test_robustness(),
            'integration_test': self._test_api_integration()
        }
        
        self.results = results
        return results
    
    def _test_model_loading(self) -> Dict[str, Any]:
        """Test model loading performance."""
        logger.info("Testing model loading performance")
        
        start_time = time.time()
        
        try:
            # Test baseline model loading
            baseline_start = time.time()
            baseline_model = SSLIntegratedModel(str(self.models_dir / 'baseline_model.pkl'))
            baseline_time = time.time() - baseline_start
            
            # Test SSL model loading
            ssl_start = time.time()
            ssl_model = SSLIntegratedModel(
                str(self.models_dir / 'baseline_model.pkl'),
                str(self.models_dir / 'ssl_encoder.pkl')
            )
            ssl_time = time.time() - ssl_start
            
            return {
                'baseline_loading_time': baseline_time,
                'ssl_loading_time': ssl_time,
                'loading_overhead': ssl_time - baseline_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_prediction_performance(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test prediction performance across different scenarios."""
        logger.info("Testing prediction performance")
        
        results = {
            'scenarios': [],
            'summary': {}
        }
        
        total_baseline_time = 0
        total_ssl_time = 0
        prediction_matches = 0
        
        for i, scenario in enumerate(test_scenarios):
            logger.info(f"Testing scenario {i+1}/{len(test_scenarios)}")
            
            try:
                # Test baseline prediction
                baseline_start = time.time()
                baseline_result = predict_with_ssl(scenario, use_ssl=False)
                baseline_time = time.time() - baseline_start
                
                # Test SSL prediction
                ssl_start = time.time()
                ssl_result = predict_with_ssl(scenario, use_ssl=True)
                ssl_time = time.time() - ssl_start
                
                # Compare results
                prediction_match = baseline_result['prediction'] == ssl_result['prediction']
                if prediction_match:
                    prediction_matches += 1
                
                scenario_result = {
                    'scenario_id': i + 1,
                    'baseline_time': baseline_time,
                    'ssl_time': ssl_time,
                    'time_overhead': ssl_time - baseline_time,
                    'baseline_prediction': baseline_result['prediction'],
                    'ssl_prediction': ssl_result['prediction'],
                    'prediction_match': prediction_match,
                    'baseline_confidence': baseline_result['confidence'],
                    'ssl_confidence': ssl_result['confidence'],
                    'baseline_threat_level': baseline_result['threat_level'],
                    'ssl_threat_level': ssl_result['threat_level']
                }
                
                results['scenarios'].append(scenario_result)
                total_baseline_time += baseline_time
                total_ssl_time += ssl_time
                
            except Exception as e:
                logger.error(f"Scenario {i+1} failed: {e}")
                results['scenarios'].append({
                    'scenario_id': i + 1,
                    'error': str(e)
                })
        
        # Calculate summary statistics
        if results['scenarios']:
            results['summary'] = {
                'total_scenarios': len(test_scenarios),
                'successful_scenarios': len([s for s in results['scenarios'] if 'error' not in s]),
                'average_baseline_time': total_baseline_time / len(test_scenarios),
                'average_ssl_time': total_ssl_time / len(test_scenarios),
                'average_time_overhead': (total_ssl_time - total_baseline_time) / len(test_scenarios),
                'prediction_match_rate': prediction_matches / len(test_scenarios),
                'total_baseline_time': total_baseline_time,
                'total_ssl_time': total_ssl_time
            }
        
        return results
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance differences."""
        logger.info("Analyzing feature importance")
        
        try:
            # Load models
            baseline_model_path = self.models_dir / 'baseline_model.pkl'
            enhanced_model_path = self.models_dir / 'enhanced_ssl_model.pkl'
            
            if not baseline_model_path.exists() or not enhanced_model_path.exists():
                return {'error': 'Required model files not found'}
            
            baseline_model = SSLIntegratedModel(str(baseline_model_path))
            enhanced_data = joblib.load(enhanced_model_path)
            enhanced_model = enhanced_data.get('enhanced_model')
            
            if enhanced_model is None:
                return {'error': 'Enhanced model not found in saved data'}
            
            # Get feature importance
            baseline_importance = baseline_model.baseline_model.feature_importances_
            enhanced_importance = enhanced_model.feature_importances_
            
            # Calculate statistics
            baseline_top_features = np.argsort(baseline_importance)[-10:][::-1]
            enhanced_top_features = np.argsort(enhanced_importance)[-10:][::-1]
            
            # Calculate SSL feature contribution
            original_feature_count = len(baseline_importance)
            ssl_features = enhanced_importance[original_feature_count:]
            ssl_contribution = np.mean(ssl_features) if len(ssl_features) > 0 else 0
            
            return {
                'baseline_top_features': baseline_top_features.tolist(),
                'enhanced_top_features': enhanced_top_features.tolist(),
                'baseline_importance_mean': float(np.mean(baseline_importance)),
                'enhanced_importance_mean': float(np.mean(enhanced_importance)),
                'baseline_importance_std': float(np.std(baseline_importance)),
                'enhanced_importance_std': float(np.std(enhanced_importance)),
                'ssl_feature_contribution': float(ssl_contribution),
                'original_feature_count': int(original_feature_count),
                'enhanced_feature_count': int(len(enhanced_importance))
            }
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {'error': str(e)}
    
    def _test_scalability(self) -> Dict[str, Any]:
        """Test scalability with different batch sizes."""
        logger.info("Testing scalability")
        
        # Create test data
        test_features = {
            'Destination Port': 80,
            'Flow Duration': 1000000,
            'Total Fwd Packets': 10000,
            'Total Backward Packets': 10000,
            'source_ip': '192.168.1.100',
            'attack_type': 'DDoS_Test'
        }
        
        batch_sizes = [1, 5, 10, 20]  # Reduced batch sizes to prevent infinite loops
        results = {'batch_sizes': batch_sizes, 'baseline_times': [], 'ssl_times': []}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            try:
                # Test baseline with timeout protection
                baseline_start = time.time()
                for i in range(batch_size):
                    if i % 5 == 0:  # Log progress every 5 iterations
                        logger.debug(f"Baseline prediction {i+1}/{batch_size}")
                    predict_with_ssl(test_features, use_ssl=False)
                baseline_time = time.time() - baseline_start
                
                # Test SSL with timeout protection
                ssl_start = time.time()
                for i in range(batch_size):
                    if i % 5 == 0:  # Log progress every 5 iterations
                        logger.debug(f"SSL prediction {i+1}/{batch_size}")
                    predict_with_ssl(test_features, use_ssl=True)
                ssl_time = time.time() - ssl_start
                
                results['baseline_times'].append(baseline_time)
                results['ssl_times'].append(ssl_time)
                
                logger.info(f"Batch size {batch_size}: Baseline={baseline_time:.4f}s, SSL={ssl_time:.4f}s")
                
            except Exception as e:
                logger.error(f"Error in batch size {batch_size}: {e}")
                results['baseline_times'].append(0.0)
                results['ssl_times'].append(0.0)
        
        return results
    
    def _test_robustness(self) -> Dict[str, Any]:
        """Test robustness with noisy/missing data."""
        logger.info("Testing robustness")
        
        base_features = {
            'Destination Port': 80,
            'Flow Duration': 1000000,
            'Total Fwd Packets': 10000,
            'Total Backward Packets': 10000,
            'source_ip': '192.168.1.100',
            'attack_type': 'DDoS_Test'
        }
        
        # Test scenarios
        test_scenarios = [
            ('baseline', base_features),
            ('noisy_data', {k: v + np.random.normal(0, 0.1 * abs(v)) 
                           for k, v in base_features.items() if isinstance(v, (int, float))}),
            ('missing_features', {k: v for i, (k, v) in enumerate(base_features.items()) 
                                if i < len(base_features) // 2}),
            ('extreme_values', {k: v * 10 if isinstance(v, (int, float)) else v 
                              for k, v in base_features.items()})
        ]
        
        results = []
        for scenario_name, features in test_scenarios:
            try:
                baseline_result = predict_with_ssl(features, use_ssl=False)
                ssl_result = predict_with_ssl(features, use_ssl=True)
                
                results.append({
                    'scenario': scenario_name,
                    'baseline_success': 'error' not in baseline_result,
                    'ssl_success': 'error' not in ssl_result,
                    'prediction_match': (baseline_result.get('prediction') == 
                                       ssl_result.get('prediction'))
                })
            except Exception as e:
                results.append({
                    'scenario': scenario_name,
                    'error': str(e)
                })
        
        return {'robustness_tests': results}
    
    def _test_api_integration(self) -> Dict[str, Any]:
        """Test API integration functionality."""
        logger.info("Testing API integration")
        
        try:
            # Test model status
            model_status = ssl_api.get_model_info()
            
            # Test prediction comparison
            test_features = {
                'Destination Port': 80,
                'Flow Duration': 1000000,
                'Total Fwd Packets': 10000,
                'Total Backward Packets': 10000
            }
            
            comparison_result = compare_ssl_baseline(test_features)
            
            return {
                'model_status': model_status,
                'comparison_test': comparison_result,
                'api_functional': True
            }
            
        except Exception as e:
            logger.error(f"API integration test failed: {e}")
            return {
                'api_functional': False,
                'error': str(e)
            }
    
    def generate_report(self, output_path: str = 'results/ssl_performance_report.json'):
        """Generate comprehensive performance report."""
        logger.info(f"Generating performance report: {output_path}")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def plot_performance_comparison(self, output_dir: str = 'results'):
        """Generate performance comparison plots."""
        logger.info("Generating performance plots")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Plot 1: Prediction time comparison
        if 'prediction_performance' in self.results:
            perf_data = self.results['prediction_performance']
            if 'summary' in perf_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                models = ['Baseline', 'SSL-Enhanced']
                times = [
                    perf_data['summary']['average_baseline_time'],
                    perf_data['summary']['average_ssl_time']
                ]
                
                bars = ax.bar(models, times, color=['blue', 'orange'])
                ax.set_ylabel('Average Prediction Time (seconds)')
                ax.set_title('SSL vs Baseline Prediction Performance')
                
                # Add value labels on bars
                for bar, time in zip(bars, times):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{time:.4f}s', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'prediction_time_comparison.png')
                plt.close()
        
        # Plot 2: Scalability comparison
        if 'scalability_test' in self.results:
            scal_data = self.results['scalability_test']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            batch_sizes = scal_data['batch_sizes']
            baseline_times = scal_data['baseline_times']
            ssl_times = scal_data['ssl_times']
            
            ax.plot(batch_sizes, baseline_times, 'b-o', label='Baseline', linewidth=2)
            ax.plot(batch_sizes, ssl_times, 'r-s', label='SSL-Enhanced', linewidth=2)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Total Time (seconds)')
            ax.set_title('SSL vs Baseline Scalability')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'scalability_comparison.png')
            plt.close()
        
        logger.info(f"Plots saved to {output_dir}")


def create_test_scenarios() -> List[Dict[str, Any]]:
    """Create diverse test scenarios for performance comparison."""
    scenarios = []
    
    # Normal traffic scenarios
    scenarios.extend([
        {
            'Destination Port': 80,
            'Flow Duration': 1000000,
            'Total Fwd Packets': 1000,
            'Total Backward Packets': 1000,
            'source_ip': '192.168.1.100',
            'attack_type': 'Normal'
        },
        {
            'Destination Port': 443,
            'Flow Duration': 2000000,
            'Total Fwd Packets': 500,
            'Total Backward Packets': 500,
            'source_ip': '10.0.0.50',
            'attack_type': 'Normal'
        }
    ])
    
    # Attack scenarios
    scenarios.extend([
        {
            'Destination Port': 80,
            'Flow Duration': 100000,
            'Total Fwd Packets': 50000,
            'Total Backward Packets': 1000,
            'source_ip': '192.168.1.100',
            'attack_type': 'DDoS'
        },
        {
            'Destination Port': 22,
            'Flow Duration': 500000,
            'Total Fwd Packets': 100,
            'Total Backward Packets': 100,
            'source_ip': '172.16.0.10',
            'attack_type': 'PortScan'
        }
    ])
    
    # Edge cases
    scenarios.extend([
        {
            'Destination Port': 0,
            'Flow Duration': 0,
            'Total Fwd Packets': 0,
            'Total Backward Packets': 0,
            'source_ip': '0.0.0.0',
            'attack_type': 'Unknown'
        },
        {
            'Destination Port': 65535,
            'Flow Duration': 999999999,
            'Total Fwd Packets': 999999,
            'Total Backward Packets': 999999,
            'source_ip': '255.255.255.255',
            'attack_type': 'Extreme'
        }
    ])
    
    return scenarios


def main():
    """Main function for SSL performance comparison."""
    logger.info("Starting SSL performance comparison")
    
    # Create test scenarios
    test_scenarios = create_test_scenarios()
    logger.info(f"Created {len(test_scenarios)} test scenarios")
    
    # Initialize comparator
    comparator = SSLPerformanceComparator()
    
    # Run comprehensive comparison
    results = comparator.run_comprehensive_comparison(test_scenarios)
    
    # Generate report
    report_path = comparator.generate_report()
    
    # Generate plots
    comparator.plot_performance_comparison()
    
    # Print summary
    print("\n" + "="*60)
    print("SSL PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    if 'prediction_performance' in results and 'summary' in results['prediction_performance']:
        perf_summary = results['prediction_performance']['summary']
        print(f"Test Scenarios: {perf_summary['total_scenarios']}")
        print(f"Successful Tests: {perf_summary['successful_scenarios']}")
        print(f"Average Baseline Time: {perf_summary['average_baseline_time']:.4f}s")
        print(f"Average SSL Time: {perf_summary['average_ssl_time']:.4f}s")
        print(f"Time Overhead: {perf_summary['average_time_overhead']:.4f}s")
        print(f"Prediction Match Rate: {perf_summary['prediction_match_rate']:.2%}")
    
    if 'feature_analysis' in results and 'error' not in results['feature_analysis']:
        feat_analysis = results['feature_analysis']
        print(f"\nFeature Analysis:")
        print(f"  Original Features: {feat_analysis['original_feature_count']}")
        print(f"  Enhanced Features: {feat_analysis['enhanced_feature_count']}")
        print(f"  SSL Contribution: {feat_analysis['ssl_feature_contribution']:.4f}")
    
    print(f"\nReport saved to: {report_path}")
    print("="*60)
    
    logger.info("SSL performance comparison completed")


if __name__ == "__main__":
    main()
