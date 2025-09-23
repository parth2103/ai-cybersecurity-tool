import numpy as np
from typing import List, Dict

class EnsembleDetector:
    def __init__(self, models: Dict):
        """
        models: Dictionary of model_name: model_instance
        """
        self.models = models
        self.weights = {}
        
    def set_weights(self, weights: Dict):
        """Set voting weights for each model"""
        self.weights = weights
        
    def predict_ensemble(self, X):
        """Combine predictions from all models"""
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                # For probabilistic models
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                # For anomaly detection models
                preds, scores = model.detect_anomalies(X)
                # Normalize scores to [0, 1]
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
                predictions[name] = 1 - normalized_scores  # Invert so higher = more anomalous
        
        # Weighted average
        final_predictions = np.zeros(len(X))
        total_weight = sum(self.weights.values())
        
        for name, preds in predictions.items():
            weight = self.weights.get(name, 1.0)
            final_predictions += preds * weight
            
        final_predictions /= total_weight
        
        return final_predictions, predictions
    
    def classify_threat_level(self, scores):
        """
        Classify threat level based on ensemble scores
        Returns: 'Critical', 'High', 'Medium', 'Low', 'None'
        """
        threat_levels = []
        
        for score in scores:
            if score >= 0.9:
                threat_levels.append('Critical')
            elif score >= 0.7:
                threat_levels.append('High')
            elif score >= 0.5:
                threat_levels.append('Medium')
            elif score >= 0.3:
                threat_levels.append('Low')
            else:
                threat_levels.append('None')
                
        return threat_levels
