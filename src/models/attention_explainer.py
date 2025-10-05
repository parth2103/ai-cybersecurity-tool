# src/models/attention_explainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionModule(nn.Module):
    """Attention mechanism for feature importance"""

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights, weights

class AttentionExplainer:
    """Main attention-based explainer"""

    def __init__(self, feature_names=None, model_path=None):
        self.feature_names = feature_names
        self.baseline_model = None
        self.attention_module = None

        if model_path:
            self.load_baseline_model(model_path)

    def load_baseline_model(self, path):
        """Load the baseline Random Forest model"""
        self.baseline_model = joblib.load(path)
        logger.info("Baseline model loaded for attention explanation")

    def compute_feature_attention(self, X, use_baseline_importance=True):
        """Compute attention weights for features"""

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        batch_size, input_dim = X.shape

        # Initialize attention module if not exists
        if self.attention_module is None:
            self.attention_module = AttentionModule(input_dim)

        # Get baseline feature importance
        if use_baseline_importance and self.baseline_model:
            baseline_importance = self.baseline_model.feature_importances_

            # Check if baseline importance matches input dimensions
            if len(baseline_importance) == input_dim:
                baseline_weights = torch.FloatTensor(baseline_importance).unsqueeze(0)
                baseline_weights = baseline_weights.expand(batch_size, -1)
            else:
                # Dimension mismatch - use uniform weights
                logger.warning(f"Baseline importance dimension ({len(baseline_importance)}) doesn't match input ({input_dim}), using uniform weights")
                baseline_weights = torch.ones(batch_size, input_dim)
        else:
            baseline_weights = torch.ones(batch_size, input_dim)

        # Compute attention weights
        self.attention_module.eval()
        with torch.no_grad():
            _, attention_weights = self.attention_module(X)

        # Combine with baseline importance
        combined_weights = attention_weights * baseline_weights
        combined_weights = F.softmax(combined_weights, dim=-1)

        return combined_weights.numpy()

    def get_top_features(self, attention_weights, top_k=5):
        """Get top k important features"""
        top_features = []

        for weights in attention_weights:
            # Get indices of top features
            top_indices = np.argsort(weights)[-top_k:][::-1]

            features = []
            for idx in top_indices:
                feature_name = (self.feature_names[idx]
                              if self.feature_names else f"Feature_{idx}")
                score = weights[idx]
                features.append((feature_name, float(score)))

            top_features.append(features)

        return top_features

    def generate_explanation(self, X, predictions=None):
        """Generate human-readable explanations"""
        attention_weights = self.compute_feature_attention(X)
        top_features = self.get_top_features(attention_weights)

        explanations = []
        for i, features in enumerate(top_features):
            # Create explanation text
            threat_level = "High" if predictions[i] > 0.7 else "Medium" if predictions[i] > 0.3 else "Low"

            explanation = {
                'prediction': float(predictions[i]) if predictions is not None else None,
                'threat_level': threat_level,
                'attention_weights': attention_weights[i].tolist(),
                'top_features': features,
                'explanation': self._create_text_explanation(features, threat_level)
            }
            explanations.append(explanation)

        return explanations

    def _create_text_explanation(self, features, threat_level):
        """Create human-readable explanation text"""
        text = f"Threat Level: {threat_level}\n"
        text += "Key indicators:\n"

        for feature_name, score in features[:3]:
            importance = "High" if score > 0.2 else "Medium" if score > 0.1 else "Low"
            text += f"  - {feature_name}: {importance} importance ({score:.3f})\n"

        return text

    def visualize_attention(self, attention_weights, feature_names=None):
        """Create visualization data for attention weights"""
        if feature_names is None:
            feature_names = self.feature_names or [f"F{i}" for i in range(len(attention_weights[0]))]

        viz_data = []
        for weights in attention_weights:
            # Sort features by attention weight
            sorted_indices = np.argsort(weights)[::-1][:10]  # Top 10

            viz_data.append({
                'features': [feature_names[i] for i in sorted_indices],
                'weights': [float(weights[i]) for i in sorted_indices]
            })

        return viz_data
