# api/endpoints/explain.py
from flask import Blueprint, request, jsonify
import numpy as np
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from models.attention_explainer import AttentionExplainer

logger = logging.getLogger(__name__)

# Create blueprint
explain_bp = Blueprint('explain', __name__, url_prefix='/explain')

# Global context (will be set by app.py)
_explainer = None
_feature_names = None
_models = None

def register_context(explainer=None, feature_names=None, models=None):
    """Register the explainer and context from main app"""
    global _explainer, _feature_names, _models
    _explainer = explainer
    _feature_names = feature_names
    _models = models
    logger.info("Explain context registered")

@explain_bp.route('', methods=['POST'])
def explain_prediction():
    """
    Explain a prediction using attention mechanism

    Request JSON:
    {
        "features": {...},  # Feature dictionary or array
        "prediction": 0.85  # Optional: provide existing prediction
    }

    Response:
    {
        "attention_weights": [...],
        "top_features": [[name, score], ...],
        "explanation": "...",
        "visualization_data": {...}
    }
    """
    try:
        if _explainer is None:
            return jsonify({"error": "Explainer not initialized"}), 503

        data = request.json or {}

        # Get features
        features = data.get('features', {})
        if isinstance(features, dict):
            # Convert dict to array based on feature_names
            if _feature_names:
                X = np.array([[features.get(fname, 0.0) for fname in _feature_names]])
            else:
                return jsonify({"error": "Feature names not available"}), 400
        elif isinstance(features, list):
            X = np.array([features])
        else:
            return jsonify({"error": "Invalid features format"}), 400

        # Get prediction if provided
        prediction = data.get('prediction')
        if prediction is None:
            # Make a prediction using the baseline model if available
            if _explainer.baseline_model is not None:
                prediction = _explainer.baseline_model.predict_proba(X)[0, 1]
            else:
                prediction = 0.5  # Default

        predictions = np.array([prediction])

        # Generate explanation
        explanations = _explainer.generate_explanation(X, predictions)

        # Extract the first explanation
        explanation_data = explanations[0] if explanations else {}

        # Get visualization data
        attention_weights = _explainer.compute_feature_attention(X)
        viz_data = _explainer.visualize_attention(attention_weights, _feature_names)

        response = {
            "success": True,
            "prediction": explanation_data.get('prediction'),
            "threat_level": explanation_data.get('threat_level'),
            "top_features": explanation_data.get('top_features', []),
            "explanation": explanation_data.get('explanation', ''),
            "visualization_data": viz_data[0] if viz_data else None,
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in explain endpoint: {str(e)}")
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500

@explain_bp.route('/batch', methods=['POST'])
def explain_batch():
    """
    Explain multiple predictions

    Request JSON:
    {
        "batch": [
            {"features": {...}, "prediction": 0.85},
            {"features": {...}, "prediction": 0.23},
            ...
        ]
    }

    Response:
    {
        "explanations": [...]
    }
    """
    try:
        if _explainer is None:
            return jsonify({"error": "Explainer not initialized"}), 503

        data = request.json or {}
        batch = data.get('batch', [])

        if not batch:
            return jsonify({"error": "Empty batch"}), 400

        # Process batch
        X_list = []
        predictions_list = []

        for item in batch:
            features = item.get('features', {})

            if isinstance(features, dict):
                if _feature_names:
                    X_list.append([features.get(fname, 0.0) for fname in _feature_names])
                else:
                    return jsonify({"error": "Feature names not available"}), 400
            elif isinstance(features, list):
                X_list.append(features)
            else:
                return jsonify({"error": "Invalid features format"}), 400

            prediction = item.get('prediction')
            if prediction is None:
                prediction = 0.5
            predictions_list.append(prediction)

        X = np.array(X_list)
        predictions = np.array(predictions_list)

        # Generate explanations
        explanations = _explainer.generate_explanation(X, predictions)

        # Get visualization data
        attention_weights = _explainer.compute_feature_attention(X)
        viz_data = _explainer.visualize_attention(attention_weights, _feature_names)

        response = {
            "success": True,
            "explanations": explanations,
            "visualization_data": viz_data,
            "count": len(explanations)
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in batch explain endpoint: {str(e)}")
        return jsonify({"error": f"Batch explanation failed: {str(e)}"}), 500

@explain_bp.route('/health', methods=['GET'])
def explain_health():
    """Check if explainer is ready"""
    explainer_ready = _explainer is not None
    feature_names_loaded = _feature_names is not None and len(_feature_names) > 0
    baseline_model_loaded = _explainer.baseline_model is not None if _explainer else False
    num_features = len(_feature_names) if _feature_names else 0

    response = {
        "status": "healthy" if all([explainer_ready, feature_names_loaded, baseline_model_loaded]) else "degraded",
        "explainer_ready": explainer_ready,
        "feature_names_loaded": feature_names_loaded,
        "baseline_model_loaded": baseline_model_loaded,
        "num_features": num_features
    }

    if response["status"] == "healthy":
        return jsonify(response)
    else:
        return jsonify(response), 503
