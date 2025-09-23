from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from queue import Queue
from pathlib import Path

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / 'models'

# Attempt to load models and preprocessors (robust to missing files)
models = {}

# Try RF baseline (support legacy name)
rf_paths = [
    MODELS_DIR / 'baseline_rf_model.pkl',
    MODELS_DIR / 'baseline_model.pkl',
]
for p in rf_paths:
    if p.exists():
        try:
            models['rf'] = joblib.load(p)
            logger.info(f"Loaded RF model from {p}")
            break
        except Exception as e:
            logger.warning(f"Failed to load RF model from {p}: {e}")

# Try XGBoost
xgb_path = MODELS_DIR / 'xgboost_model.pkl'
if xgb_path.exists():
    try:
        models['xgboost'] = joblib.load(xgb_path)
        logger.info(f"Loaded XGBoost model from {xgb_path}")
    except Exception as e:
        logger.warning(f"Failed to load XGBoost model: {e}")

# Try Isolation Forest
if_path = MODELS_DIR / 'isolation_forest.pkl'
if if_path.exists():
    try:
        models['isolation_forest'] = joblib.load(if_path)
        logger.info(f"Loaded Isolation Forest model from {if_path}")
    except Exception as e:
        logger.warning(f"Failed to load Isolation Forest model: {e}")

# Scaler
scaler = None
scaler_path = MODELS_DIR / 'scaler.pkl'
if scaler_path.exists():
    try:
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler.pkl")
    except Exception as e:
        logger.warning(f"Failed to load scaler.pkl: {e}")

# Feature names (used as selected features)
selected_features = []
feat_paths = [MODELS_DIR / 'selected_features.pkl', MODELS_DIR / 'feature_names.pkl']
for p in feat_paths:
    if p.exists():
        try:
            selected_features = list(joblib.load(p))
            logger.info(f"Loaded feature list from {p}")
            break
        except Exception as e:
            logger.warning(f"Failed to load feature list from {p}: {e}")

# Label encoder (optional)
label_encoder = None
le_path = MODELS_DIR / 'label_encoder.pkl'
if le_path.exists():
    try:
        label_encoder = joblib.load(le_path)
        logger.info("Loaded label_encoder.pkl")
    except Exception as e:
        logger.warning(f"Failed to load label_encoder.pkl: {e}")

# Alert queue for real-time notifications
alert_queue = Queue()

# Threat statistics
threat_stats = {
    'total_requests': 0,
    'threats_detected': 0,
    'threat_history': [],
    'current_threat_level': 'Low'
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if not models:
            return jsonify({'error': 'No models loaded'}), 503

        data = request.json or {}
        # Extract features
        features = data.get('features', {})
        
        # Convert to DataFrame for consistency
        df = pd.DataFrame([features])
        
        # Ensure we have all required features
        for feature in selected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select and order features if we have a feature list
        if selected_features:
            df = df[selected_features]
        
        X = df.values
        # Scale features if scaler available
        if scaler is not None:
            X = scaler.transform(df)
        
        # Get predictions from all models
        predictions = {}
        threat_scores = []
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = float(model.predict_proba(X)[0, 1])
                    predictions[model_name] = prob
                    threat_scores.append(prob)
                else:
                    # For anomaly detection
                    pred = model.predict(X)[0]
                    score = 1.0 if int(pred) == -1 else 0.0
                    predictions[model_name] = score
                    threat_scores.append(score)
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")
        
        if not threat_scores:
            return jsonify({'error': 'No predictions available'}), 500
        
        # Calculate average threat score
        avg_threat_score = float(np.mean(threat_scores))
        
        # Determine threat level
        if avg_threat_score >= 0.8:
            threat_level = 'Critical'
        elif avg_threat_score >= 0.6:
            threat_level = 'High'
        elif avg_threat_score >= 0.4:
            threat_level = 'Medium'
        elif avg_threat_score >= 0.2:
            threat_level = 'Low'
        else:
            threat_level = 'None'
        
        # Update statistics
        threat_stats['total_requests'] += 1
        if threat_level in ['Critical', 'High']:
            threat_stats['threats_detected'] += 1
            
            # Create alert
            alert = {
                'timestamp': datetime.now().isoformat(),
                'threat_level': threat_level,
                'threat_score': avg_threat_score,
                'source_ip': data.get('source_ip', 'Unknown'),
                'attack_type': data.get('attack_type', 'Unknown')
            }
            
            # Add to alert queue and emit
            alert_queue.put(alert)
            socketio.emit('new_alert', alert)
        
        # Add to history
        threat_stats['threat_history'].append({
            'timestamp': datetime.now().isoformat(),
            'threat_score': avg_threat_score,
            'threat_level': threat_level
        })
        
        # Keep only last 100 entries
        if len(threat_stats['threat_history']) > 100:
            threat_stats['threat_history'] = threat_stats['threat_history'][-100:]
        
        threat_stats['current_threat_level'] = threat_level
        
        response = {
            'threat_detected': threat_level in ['Critical', 'High', 'Medium'],
            'threat_level': threat_level,
            'threat_score': avg_threat_score,
            'model_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify(threat_stats)

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    alerts = []
    while not alert_queue.empty() and len(alerts) < 50:
        alerts.append(alert_queue.get())
    
    return jsonify({'alerts': alerts})

@app.route('/system/info', methods=['GET'])
def system_info():
    """Get system information"""
    import psutil
    
    info = {
        'models_loaded': list(models.keys()),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'total_predictions': threat_stats['total_requests'],
        'threats_detected': threat_stats['threats_detected'],
        'detection_rate': (threat_stats['threats_detected'] / max(threat_stats['total_requests'], 1)) * 100
    }
    
    return jsonify(info)

# WebSocket events
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'data': 'Connected to threat detection system'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001, allow_unsafe_werkzeug=True)
