from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .app import models, scaler, selected_features

batch_bp = Blueprint('batch', __name__)

# Thread pool for future async work (not strictly necessary now)
executor = ThreadPoolExecutor(max_workers=4)


def preprocess_batch(df: pd.DataFrame) -> np.ndarray:
    # Ensure all required features exist
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0
    # Select and order
    if selected_features:
        df = df[selected_features]
    X = df.values
    # Scale if scaler is available
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
        
        # Use RF if available, otherwise first available model with predict_proba/predict
        model = models.get('rf')
        if model is None:
            # fallback: pick first model
            model = next(iter(models.values()))
        
        # Get predictions
        has_proba = hasattr(model, 'predict_proba')
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if has_proba else None
        
        results = []
        for i in range(len(df)):
            if has_proba:
                prob = probabilities[i, 1] if probabilities.shape[1] > 1 else probabilities[i, 0]
                threat_score = float(prob)
                threat_detected = bool(predictions[i])
            else:
                pred = predictions[i]
                # IsolationForest style: -1 anomaly -> score 1.0
                threat_score = float(1.0 if int(pred) == -1 else 0.0)
                threat_detected = (int(pred) == -1)
            
            results.append({
                'index': i,
                'threat_detected': threat_detected,
                'threat_score': threat_score,
                'threat_level': classify_threat_level(threat_score)
            })
        
        # Summary statistics
        summary = {
            'total_processed': len(results),
            'threats_detected': sum(1 for r in results if r['threat_detected']),
            'average_threat_score': float(np.mean([r['threat_score'] for r in results])) if results else 0.0,
            'critical_threats': sum(1 for r in results if r['threat_level'] == 'Critical')
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
