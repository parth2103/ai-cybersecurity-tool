"""
SSL API Integration Module

This module provides API-compatible functions for SSL-enhanced predictions
without breaking existing production code.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple
import time
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

from .ssl_enhancement import SSLEnhancement
from .integrate_ssl import SSLIntegratedModel

logger = logging.getLogger(__name__)


class SSLAPIIntegration:
    """API-compatible SSL integration for production use."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize SSL API integration.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.ssl_encoder = None
        self.enhanced_model = None
        self.baseline_model = None
        self.feature_scaler = None
        self.feature_names = None
        
        # Load available models
        self._load_models()
    
    def _load_models(self):
        """Load all available models and components."""
        try:
            # Load baseline model
            baseline_path = self.models_dir / 'baseline_model.pkl'
            if baseline_path.exists():
                self.baseline_model = joblib.load(baseline_path)
                logger.info("Loaded baseline Random Forest model")
            
            # Load SSL encoder
            ssl_encoder_path = self.models_dir / 'ssl_encoder.pkl'
            if ssl_encoder_path.exists():
                self.ssl_encoder = SSLEnhancement.load_encoder(str(ssl_encoder_path))
                logger.info("Loaded SSL encoder")
            
            # Load enhanced model
            enhanced_path = self.models_dir / 'enhanced_ssl_model.pkl'
            if enhanced_path.exists():
                enhanced_data = joblib.load(enhanced_path)
                self.enhanced_model = enhanced_data.get('enhanced_model')
                logger.info("Loaded SSL-enhanced model")
            
            # Load feature scaler and names
            scaler_path = self.models_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            
            feature_names_path = self.models_dir / 'feature_names.pkl'
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
                logger.info("Loaded feature names")
            
        except Exception as e:
            logger.warning(f"Some models could not be loaded: {e}")
    
    def predict_threat(self, features: Dict[str, Any], use_ssl: bool = True) -> Dict[str, Any]:
        """
        Predict threat using either baseline or SSL-enhanced model.
        
        Args:
            features: Network traffic features dictionary
            use_ssl: Whether to use SSL-enhanced prediction
        
        Returns:
            Prediction results with metadata
        """
        start_time = time.time()
        
        try:
            # Convert features to array
            X = self._prepare_features(features)
            
            # Choose model
            if use_ssl and self.enhanced_model is not None and self.ssl_encoder is not None:
                # Use SSL-enhanced model
                prediction, confidence = self._predict_enhanced(X)
                model_type = "ssl_enhanced"
            elif self.baseline_model is not None:
                # Use baseline model
                prediction, confidence = self._predict_baseline(X)
                model_type = "baseline"
            else:
                raise ValueError("No trained models available")
            
            # Calculate threat level
            threat_level = self._calculate_threat_level(confidence)
            
            inference_time = time.time() - start_time
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'threat_level': threat_level,
                'model_type': model_type,
                'inference_time': inference_time,
                'features_used': X.shape[1],
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'error': str(e),
                'prediction': 'unknown',
                'confidence': 0.0,
                'threat_level': 'unknown',
                'model_type': 'error',
                'inference_time': time.time() - start_time
            }
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction."""
        if self.feature_names is not None:
            # Use known feature names
            feature_array = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_array.append(float(features[feature_name]))
                else:
                    # Handle missing features
                    feature_array.append(0.0)
            X = np.array(feature_array).reshape(1, -1)
        else:
            # Use all provided features
            X = np.array(list(features.values())).reshape(1, -1)
        
        # Apply scaling if available
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        
        return X
    
    def _predict_baseline(self, X: np.ndarray) -> Tuple[str, float]:
        """Predict using baseline model."""
        prediction = self.baseline_model.predict(X)[0]
        probabilities = self.baseline_model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        # Convert prediction to string
        prediction_str = str(prediction)
        return prediction_str, confidence
    
    def _predict_enhanced(self, X: np.ndarray) -> Tuple[str, float]:
        """Predict using SSL-enhanced model."""
        # Create enhanced features
        if self.ssl_encoder is not None:
            ssl_features = self.ssl_encoder.encode_features(X)
            X_enhanced = np.concatenate([X, ssl_features], axis=1)
        else:
            X_enhanced = X
        
        prediction = self.enhanced_model.predict(X_enhanced)[0]
        probabilities = self.enhanced_model.predict_proba(X_enhanced)[0]
        confidence = float(np.max(probabilities))
        
        # Convert prediction to string
        prediction_str = str(prediction)
        return prediction_str, confidence
    
    def _calculate_threat_level(self, confidence: float) -> str:
        """Calculate threat level based on confidence."""
        if confidence < 0.3:
            return "Low"
        elif confidence < 0.6:
            return "Medium"
        elif confidence < 0.8:
            return "High"
        else:
            return "Critical"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'baseline_model_loaded': self.baseline_model is not None,
            'ssl_encoder_loaded': self.ssl_encoder is not None,
            'enhanced_model_loaded': self.enhanced_model is not None,
            'feature_scaler_loaded': self.feature_scaler is not None,
            'feature_names_loaded': self.feature_names is not None,
            'ssl_available': self.ssl_encoder is not None and self.enhanced_model is not None
        }
    
    def compare_predictions(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compare predictions between baseline and SSL-enhanced models."""
        baseline_result = self.predict_threat(features, use_ssl=False)
        enhanced_result = self.predict_threat(features, use_ssl=True)
        
        return {
            'baseline': baseline_result,
            'enhanced': enhanced_result,
            'prediction_match': baseline_result['prediction'] == enhanced_result['prediction'],
            'confidence_difference': enhanced_result['confidence'] - baseline_result['confidence'],
            'threat_level_match': baseline_result['threat_level'] == enhanced_result['threat_level']
        }


def create_ssl_api_wrapper():
    """Create a global SSL API wrapper instance."""
    return SSLAPIIntegration()


# Global instance for API use
ssl_api = create_ssl_api_wrapper()


def predict_with_ssl(features: Dict[str, Any], use_ssl: bool = True) -> Dict[str, Any]:
    """
    Convenience function for SSL-enhanced predictions.
    
    Args:
        features: Network traffic features
        use_ssl: Whether to use SSL enhancement
    
    Returns:
        Prediction results
    """
    return ssl_api.predict_threat(features, use_ssl=use_ssl)


def compare_ssl_baseline(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare SSL-enhanced vs baseline predictions.
    
    Args:
        features: Network traffic features
    
    Returns:
        Comparison results
    """
    return ssl_api.compare_predictions(features)


def get_ssl_model_status() -> Dict[str, Any]:
    """Get status of SSL models."""
    return ssl_api.get_model_info()


if __name__ == "__main__":
    # Test the SSL API integration
    test_features = {
        'Destination Port': 80,
        'Flow Duration': 1000000,
        'Total Fwd Packets': 10000,
        'Total Backward Packets': 10000,
        'source_ip': '192.168.1.100',
        'attack_type': 'DDoS_Test'
    }
    
    print("Testing SSL API Integration...")
    print(f"Model Status: {get_ssl_model_status()}")
    
    # Test prediction
    result = predict_with_ssl(test_features, use_ssl=True)
    print(f"SSL Prediction: {result}")
    
    # Test comparison
    comparison = compare_ssl_baseline(test_features)
    print(f"Comparison: {comparison}")
