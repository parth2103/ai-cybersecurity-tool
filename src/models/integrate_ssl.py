"""
SSL Integration Module for Cybersecurity Detection

This module integrates self-supervised learning features with existing models
to enhance performance without breaking production code.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional
import time

from .ssl_enhancement import SSLEnhancement, create_ssl_enhanced_features
from data_loader import CICIDSDataLoader
from preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSLIntegratedModel:
    """Enhanced model that combines original features with SSL-learned representations."""
    
    def __init__(self, baseline_model_path: str, ssl_encoder_path: Optional[str] = None):
        """
        Args:
            baseline_model_path: Path to existing baseline model
            ssl_encoder_path: Path to trained SSL encoder (optional)
        """
        self.baseline_model_path = baseline_model_path
        self.ssl_encoder_path = ssl_encoder_path
        
        # Load baseline model
        self.baseline_model = self._load_baseline_model()
        
        # Enhanced model (will be trained with SSL features)
        self.enhanced_model = None
        
        # Feature scaler for SSL features
        self.ssl_scaler = None
        
        # Performance metrics
        self.performance_comparison = {}
        
        logger.info(f"Loaded baseline model from {baseline_model_path}")
    
    def _load_baseline_model(self):
        """Load the existing baseline Random Forest model."""
        try:
            model = joblib.load(self.baseline_model_path)
            logger.info("Successfully loaded baseline Random Forest model")
            return model
        except Exception as e:
            logger.error(f"Failed to load baseline model: {e}")
            raise
    
    def prepare_ssl_encoder(self, X_unlabeled: np.ndarray, 
                           epochs: int = 100, batch_size: int = 256,
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train SSL encoder on unlabeled data if not already available.
        
        Args:
            X_unlabeled: Unlabeled network traffic data for SSL pretraining
            epochs: Number of training epochs
            batch_size: Batch size for training
            save_path: Path to save the trained encoder
        
        Returns:
            Training metrics
        """
        if self.ssl_encoder_path and Path(self.ssl_encoder_path).exists():
            logger.info("SSL encoder already exists, skipping pretraining")
            return {"status": "encoder_exists"}
        
        logger.info("Training SSL encoder on unlabeled data")
        
        # Initialize SSL enhancement
        ssl = SSLEnhancement(input_dim=X_unlabeled.shape[1])
        
        # Train encoder
        save_path = save_path or 'models/ssl_encoder.pkl'
        metrics = ssl.pretrain(X_unlabeled, epochs=epochs, batch_size=batch_size, 
                              save_path=save_path)
        
        self.ssl_encoder_path = save_path
        logger.info(f"SSL encoder trained and saved to {save_path}")
        
        return metrics
    
    def create_enhanced_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create both original and SSL-enhanced features.
        
        Args:
            X: Input features
        
        Returns:
            Tuple of (original_features, enhanced_features)
        """
        # Original features (as used by baseline model)
        original_features = X.copy()
        
        if self.ssl_encoder_path and Path(self.ssl_encoder_path).exists():
            # Create SSL-enhanced features
            enhanced_features = create_ssl_enhanced_features(X, self.ssl_encoder_path)
            logger.info(f"Created enhanced features: {enhanced_features.shape}")
        else:
            logger.warning("SSL encoder not available, using original features only")
            enhanced_features = original_features
        
        return original_features, enhanced_features
    
    def train_enhanced_model(self, X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train enhanced model with SSL features and compare performance.
        
        Args:
            X: Training features
            y: Training labels
            test_size: Fraction of data for testing
            random_state: Random state for reproducibility
        
        Returns:
            Performance comparison metrics
        """
        logger.info("Training enhanced model with SSL features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create features
        X_train_orig, X_train_enhanced = self.create_enhanced_features(X_train)
        X_test_orig, X_test_enhanced = self.create_enhanced_features(X_test)
        
        # Clean enhanced features - handle infinity and NaN values
        X_train_enhanced = np.nan_to_num(X_train_enhanced, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_enhanced = np.nan_to_num(X_test_enhanced, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Also clean original features for baseline model
        X_train_orig = np.nan_to_num(X_train_orig, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_orig = np.nan_to_num(X_test_orig, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Train enhanced Random Forest model
        enhanced_model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        start_time = time.time()
        enhanced_model.fit(X_train_enhanced, y_train)
        training_time = time.time() - start_time
        
        # Evaluate models
        results = self._evaluate_models(
            X_test_orig, X_test_enhanced, y_test, 
            training_time, enhanced_model
        )
        
        self.enhanced_model = enhanced_model
        self.performance_comparison = results
        
        return results
    
    def _evaluate_models(self, X_test_orig: np.ndarray, X_test_enhanced: np.ndarray,
                        y_test: np.ndarray, training_time: float, 
                        enhanced_model: RandomForestClassifier) -> Dict[str, Any]:
        """Evaluate and compare baseline vs enhanced model performance."""
        
        # Baseline model predictions
        baseline_pred = self.baseline_model.predict(X_test_orig)
        baseline_acc = accuracy_score(y_test, baseline_pred)
        
        # Enhanced model predictions
        enhanced_pred = enhanced_model.predict(X_test_enhanced)
        enhanced_acc = accuracy_score(y_test, enhanced_pred)
        
        # Inference time comparison
        start_time = time.time()
        _ = self.baseline_model.predict(X_test_orig)
        baseline_inference_time = time.time() - start_time
        
        start_time = time.time()
        _ = enhanced_model.predict(X_test_enhanced)
        enhanced_inference_time = time.time() - start_time
        
        # Feature importance analysis
        baseline_importance = self.baseline_model.feature_importances_
        enhanced_importance = enhanced_model.feature_importances_
        
        results = {
            'baseline_accuracy': baseline_acc,
            'enhanced_accuracy': enhanced_acc,
            'accuracy_improvement': enhanced_acc - baseline_acc,
            'relative_improvement': ((enhanced_acc - baseline_acc) / baseline_acc) * 100,
            'baseline_inference_time': baseline_inference_time,
            'enhanced_inference_time': enhanced_inference_time,
            'enhanced_training_time': training_time,
            'baseline_features': X_test_orig.shape[1],
            'enhanced_features': X_test_enhanced.shape[1],
            'baseline_importance_mean': np.mean(baseline_importance),
            'enhanced_importance_mean': np.mean(enhanced_importance),
            'baseline_importance_std': np.std(baseline_importance),
            'enhanced_importance_std': np.std(enhanced_importance)
        }
        
        # Detailed classification reports
        results['baseline_classification_report'] = classification_report(
            y_test, baseline_pred, output_dict=True
        )
        results['enhanced_classification_report'] = classification_report(
            y_test, enhanced_pred, output_dict=True
        )
        
        # Confusion matrices
        results['baseline_confusion_matrix'] = confusion_matrix(y_test, baseline_pred).tolist()
        results['enhanced_confusion_matrix'] = confusion_matrix(y_test, enhanced_pred).tolist()
        
        logger.info(f"Performance Comparison:")
        logger.info(f"  Baseline Accuracy: {baseline_acc:.4f}")
        logger.info(f"  Enhanced Accuracy: {enhanced_acc:.4f}")
        logger.info(f"  Improvement: {results['relative_improvement']:.2f}%")
        logger.info(f"  Baseline Inference Time: {baseline_inference_time:.4f}s")
        logger.info(f"  Enhanced Inference Time: {enhanced_inference_time:.4f}s")
        
        return results
    
    def predict(self, X: np.ndarray, use_enhanced: bool = True) -> np.ndarray:
        """
        Make predictions using either baseline or enhanced model.
        
        Args:
            X: Input features
            use_enhanced: Whether to use enhanced model with SSL features
        
        Returns:
            Predictions
        """
        if use_enhanced and self.enhanced_model is not None:
            _, X_enhanced = self.create_enhanced_features(X)
            return self.enhanced_model.predict(X_enhanced)
        else:
            return self.baseline_model.predict(X)
    
    def predict_proba(self, X: np.ndarray, use_enhanced: bool = True) -> np.ndarray:
        """Get prediction probabilities."""
        if use_enhanced and self.enhanced_model is not None:
            _, X_enhanced = self.create_enhanced_features(X)
            return self.enhanced_model.predict_proba(X_enhanced)
        else:
            return self.baseline_model.predict_proba(X)
    
    def save_enhanced_model(self, path: str):
        """Save the enhanced model and metadata."""
        save_data = {
            'enhanced_model': self.enhanced_model,
            'ssl_encoder_path': self.ssl_encoder_path,
            'baseline_model_path': self.baseline_model_path,
            'performance_comparison': self.performance_comparison
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_data, path)
        logger.info(f"Enhanced model saved to {path}")
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Analyze feature importance differences between models."""
        if not self.performance_comparison:
            raise ValueError("No performance comparison available. Train the enhanced model first.")
        
        baseline_importance = self.baseline_model.feature_importances_
        enhanced_importance = self.enhanced_model.feature_importances_
        
        # Find most important features for each model
        baseline_top_features = np.argsort(baseline_importance)[-10:][::-1]
        enhanced_top_features = np.argsort(enhanced_importance)[-10:][::-1]
        
        analysis = {
            'baseline_top_features': baseline_top_features.tolist(),
            'enhanced_top_features': enhanced_top_features.tolist(),
            'baseline_top_importance': baseline_importance[baseline_top_features].tolist(),
            'enhanced_top_importance': enhanced_importance[enhanced_top_features].tolist(),
            'ssl_feature_contribution': np.mean(enhanced_importance[self.baseline_model.n_features_in_:])
        }
        
        return analysis


def run_ssl_integration_pipeline(data_sample_size: int = 10000, 
                                ssl_epochs: int = 50,
                                test_ssl_only: bool = False,
                                X_processed: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Complete SSL integration pipeline.
    
    Args:
        data_sample_size: Number of samples to use for training
        ssl_epochs: Number of epochs for SSL pretraining
        test_ssl_only: If True, only test SSL without full integration
        X_processed: Pre-processed features (for SSL-only mode)
    
    Returns:
        Complete pipeline results
    """
    logger.info("Starting SSL integration pipeline")
    
    if test_ssl_only and X_processed is not None:
        # SSL-only mode with pre-processed features
        logger.info("Using pre-processed features for SSL-only mode")
        X = X_processed
    else:
        # Full integration mode - load and process data with labels
        loader = CICIDSDataLoader()
        df = loader.load_friday_data(sample_size=data_sample_size)
        
        # Prepare features and labels
        feature_cols = [col for col in df.columns if col not in ['Label', ' Label']]
        label_col = ' Label' if ' Label' in df.columns else 'Label'
        
        X = df[feature_cols].values
        y = df[label_col].values
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        # For SSL integration, we need both features and labels
        # Create a temporary DataFrame with features and labels
        df_temp = df[feature_cols + [label_col]].copy()
        
        # Load preprocessor and transform features
        preprocessor = DataPreprocessor()
        X_processed, y_processed = preprocessor.prepare_features(df_temp)
    
    # Initialize integrated model
    baseline_model_path = 'models/baseline_model.pkl'
    integrated_model = SSLIntegratedModel(baseline_model_path)
    
    # Train SSL encoder
    ssl_metrics = integrated_model.prepare_ssl_encoder(
        X, epochs=ssl_epochs, batch_size=256
    )
    
    if test_ssl_only:
        return {
            'ssl_training': ssl_metrics,
            'status': 'ssl_only_test'
        }
    
    # Train enhanced model and compare performance
    performance_results = integrated_model.train_enhanced_model(X_processed, y_processed)
    
    # Feature importance analysis
    feature_analysis = integrated_model.get_feature_importance_analysis()
    
    # Save enhanced model
    integrated_model.save_enhanced_model('models/enhanced_ssl_model.pkl')
    
    # Compile results
    results = {
        'ssl_training': ssl_metrics,
        'performance_comparison': performance_results,
        'feature_analysis': feature_analysis,
        'data_info': {
            'total_samples': int(len(X_processed)),
            'features': int(X_processed.shape[1]),
            'classes': int(len(np.unique(y_processed))),
            'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(y_processed, return_counts=True))}
        }
    }
    
    logger.info("SSL integration pipeline completed successfully")
    return results


if __name__ == "__main__":
    # Run the complete integration pipeline
    results = run_ssl_integration_pipeline(
        data_sample_size=5000,
        ssl_epochs=30,
        test_ssl_only=False
    )
    
    print("\n" + "="*50)
    print("SSL INTEGRATION RESULTS")
    print("="*50)
    
    # Print performance comparison
    perf = results['performance_comparison']
    print(f"Baseline Accuracy: {perf['baseline_accuracy']:.4f}")
    print(f"Enhanced Accuracy: {perf['enhanced_accuracy']:.4f}")
    print(f"Improvement: {perf['relative_improvement']:.2f}%")
    print(f"Feature Count - Baseline: {perf['baseline_features']}, Enhanced: {perf['enhanced_features']}")
    
    # Print feature analysis
    feat_analysis = results['feature_analysis']
    print(f"\nSSL Feature Contribution: {feat_analysis['ssl_feature_contribution']:.4f}")
    
    print("\nPipeline completed successfully!")
