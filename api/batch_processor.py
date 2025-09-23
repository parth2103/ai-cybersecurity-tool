from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Reuse objects from main app
try:
    from .app import models, scaler, selected_features
except ImportError:
    from app import models, scaler, selected_features

batch_bp = Blueprint('batch', __name__)

executor = ThreadPoolExecutor(max_workers=4)


def preprocess_batch(df: pd.DataFrame) -> np.ndarray:
    """Align columns to selected_features and apply scaler if available."""
    if selected_features:
        for feature in selected_features:
            if feature not in df.columns:
                df[feature] = 0
        df = df[selected_features]
    X = df.values
    if scaler is not None:
        X = scaler.transform(df)
    return X


@batch_bp.route('/batch/predict', methods=['POST'])
def batch_predict():
    """Process multiple logs at once"""
    try:
        data = request.json or {}
        logs = data.get('logs', [])
        
        if not logs:
            return jsonify({'error': 'No logs provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(logs)
        
        # Preprocess all at once
        X = preprocess_batch(df)
        
        # Pick a default model (rf) if available
        if 'rf' not in models:
            return jsonify({'error': 'RF model not loaded'}), 503
        model = models['rf']
        
        # Get predictions (use executor.map if heavy)
        predictions = model.predict(X)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
        
        results = []
        for i in range(len(predictions)):
            pred = predictions[i]
            if probabilities is not None:
                prob = probabilities[i]
                threat_score = float(prob[1]) if len(prob) > 1 else float(pred)
            else:
                threat_score = float(pred)
            results.append({
                'index': i,
                'threat_detected': bool(pred),
                'threat_score': threat_score,
                'threat_level': classify_threat_level(threat_score)
            })
        
        # Summary statistics
        summary = {
            'total_processed': len(results),
            'threats_detected': int(sum(1 for r in results if r['threat_detected'])),
            'average_threat_score': float(np.mean([r['threat_score'] for r in results])),
            'critical_threats': int(sum(1 for r in results if r['threat_level'] == 'Critical'))
        }
        
        return jsonify({
            'results': results,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def classify_threat_level(score: float) -> str:
    """Classify threat level based on score"""
    if score >= 0.9:
        return 'Critical'
    elif score >= 0.7:
        return 'High'
    elif score >= 0.5:
        return 'Medium'
    elif score >= 0.3:
        return 'Low'
    else:
        return 'None'
