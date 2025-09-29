#!/usr/bin/env python3
"""
SSL-Enhanced Cybersecurity Model Training Script

This script demonstrates the complete pipeline for enhancing the existing
Random Forest model with self-supervised learning features.
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import json
import time
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from models.integrate_ssl import run_ssl_integration_pipeline
from data_loader import CICIDSDataLoader
from preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ssl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_environment():
    """Validate that the environment is set up correctly."""
    logger.info("Validating environment...")
    
    # Check if baseline model exists
    baseline_model_path = Path('models/baseline_model.pkl')
    if not baseline_model_path.exists():
        logger.error(f"Baseline model not found at {baseline_model_path}")
        logger.error("Please run the baseline training first: python run_week1.py")
        return False
    
    # Check if data directory exists
    data_dir = Path('data/cicids2017/MachineLearningCCSV')
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        logger.warning("Will use synthetic data for demonstration")
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    logger.info("Environment validation completed")
    return True


def run_ssl_pretraining_only(data_sample_size: int = 10000, epochs: int = 50):
    """Run only SSL pretraining without full integration."""
    logger.info("Running SSL pretraining only...")
    
    # Load data
    loader = CICIDSDataLoader()
    df = loader.load_friday_data(sample_size=data_sample_size)
    
    # Prepare features (exclude label columns for SSL pretraining)
    feature_cols = [col for col in df.columns if col not in ['Label', ' Label']]
    X = df[feature_cols].values
    
    # For SSL pretraining, we only need the features (no labels)
    # Clean the data manually since prepare_features expects labels
    X_df = pd.DataFrame(X, columns=feature_cols)
    
    # Clean the data (handle infinity and NaN)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0)
    
    # Select only numeric columns
    X_processed = X_df.select_dtypes(include=[np.number]).values
    
    logger.info(f"SSL pretraining data: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
    
    # Run SSL pretraining
    results = run_ssl_integration_pipeline(
        data_sample_size=data_sample_size,
        ssl_epochs=epochs,
        test_ssl_only=True,
        X_processed=X_processed
    )
    
    return results


def run_full_ssl_pipeline(data_sample_size: int = 10000, ssl_epochs: int = 50):
    """Run the complete SSL integration pipeline."""
    logger.info("Running full SSL integration pipeline...")
    
    start_time = time.time()
    
    # Run complete pipeline
    results = run_ssl_integration_pipeline(
        data_sample_size=data_sample_size,
        ssl_epochs=ssl_epochs,
        test_ssl_only=False
    )
    
    total_time = time.time() - start_time
    results['total_pipeline_time'] = total_time
    
    # Save results
    results_path = 'results/ssl_integration_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Total pipeline time: {total_time:.2f} seconds")
    
    return results


def print_results_summary(results: dict):
    """Print a summary of the SSL integration results."""
    print("\n" + "="*60)
    print("SSL ENHANCEMENT RESULTS SUMMARY")
    print("="*60)
    
    # Data information
    data_info = results.get('data_info', {})
    print(f"Dataset: {data_info.get('total_samples', 'N/A')} samples, "
          f"{data_info.get('features', 'N/A')} features")
    
    # SSL training results
    ssl_training = results.get('ssl_training', {})
    if 'final_train_loss' in ssl_training:
        print(f"SSL Training: {ssl_training.get('epochs_trained', 'N/A')} epochs, "
              f"Final Loss: {ssl_training.get('final_train_loss', 'N/A'):.4f}")
    
    # Performance comparison
    perf = results.get('performance_comparison', {})
    if perf:
        print(f"\nPerformance Comparison:")
        print(f"  Baseline Accuracy: {perf.get('baseline_accuracy', 'N/A'):.4f}")
        print(f"  Enhanced Accuracy: {perf.get('enhanced_accuracy', 'N/A'):.4f}")
        print(f"  Improvement: {perf.get('relative_improvement', 'N/A'):.2f}%")
        print(f"  Features: {perf.get('baseline_features', 'N/A')} → {perf.get('enhanced_features', 'N/A')}")
        
        # Inference time comparison
        baseline_time = perf.get('baseline_inference_time', 0)
        enhanced_time = perf.get('enhanced_inference_time', 0)
        if baseline_time > 0:
            print(f"  Inference Time: {baseline_time:.4f}s → {enhanced_time:.4f}s")
    
    # Feature analysis
    feat_analysis = results.get('feature_analysis', {})
    if feat_analysis:
        ssl_contribution = feat_analysis.get('ssl_feature_contribution', 0)
        print(f"\nFeature Analysis:")
        print(f"  SSL Feature Contribution: {ssl_contribution:.4f}")
    
    # Pipeline timing
    if 'total_pipeline_time' in results:
        print(f"\nPipeline completed in {results['total_pipeline_time']:.2f} seconds")
    
    print("="*60)


def main():
    """Main function to run SSL enhancement training."""
    parser = argparse.ArgumentParser(description='Train SSL-enhanced cybersecurity model')
    parser.add_argument('--mode', choices=['ssl-only', 'full'], default='full',
                       help='Training mode: ssl-only or full integration')
    parser.add_argument('--data-size', type=int, default=10000,
                       help='Number of samples to use for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for SSL pretraining')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting SSL enhancement training")
    logger.info(f"Mode: {args.mode}, Data size: {args.data_size}, Epochs: {args.epochs}")
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    try:
        if args.mode == 'ssl-only':
            results = run_ssl_pretraining_only(args.data_size, args.epochs)
            logger.info("SSL pretraining completed")
        else:
            results = run_full_ssl_pipeline(args.data_size, args.epochs)
            logger.info("Full SSL integration pipeline completed")
        
        # Print results summary
        print_results_summary(results)
        
        logger.info("SSL enhancement training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
