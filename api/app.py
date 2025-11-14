from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from queue import Queue
from pathlib import Path
import time
import sys

# Add src to path for imports
sys.path.append("src")

from utils.logger import (
    get_logger,
    log_threat,
    log_api_request,
    log_health,
    log_security,
)
from utils.error_handler import (
    handle_errors,
    validate_data,
    validate_model_input,
    safe_model_prediction,
    handle_api_error,
    handle_model_error,
    ErrorSeverity,
    ErrorCategory,
    SecurityError,
    ModelError,
    DataValidationError,
    get_error_summary,
)
from utils.data_validator import validate_request, prepare_features, detect_anomalies
from utils.auth import (
    require_api_key,
    require_permission,
    security_check,
    get_current_user,
    create_api_key,
    revoke_api_key,
    get_api_key_info,
    api_key_manager,
)
from utils.database import (
    store_threat,
    store_alert,
    store_system_metrics,
    store_api_usage,
    get_threats,
    get_alerts,
    get_threat_statistics,
    threat_db,
)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure enhanced logging
logger = get_logger("api")

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# Attempt to load models and preprocessors (robust to missing files)
# Priority: New models (trained on IoT-IDAD 2024 + CICAPT-IIOT) > Old models (CICIDS2017)
models = {}

# Try RF baseline - prefer new models trained on new datasets
rf_paths = [
    MODELS_DIR / "random_forest_new_datasets.pkl",  # New model (IoT-IDAD + CICAPT)
    MODELS_DIR / "baseline_model.pkl",  # Fallback to old model
]
for p in rf_paths:
    if p.exists():
        try:
            models["rf"] = joblib.load(p)
            model_type = "NEW (IoT-IDAD + CICAPT)" if "new_datasets" in str(p) else "OLD (CICIDS2017)"
            logger.info(f"Loaded RF model from {p} [{model_type}]")
            break
        except Exception as e:
            logger.warning(f"Failed to load RF model from {p}: {e}")

# Try XGBoost - prefer new models
xgb_paths = [
    MODELS_DIR / "xgboost_model_new_datasets.pkl",  # New model
    MODELS_DIR / "xgboost_model.pkl",  # Fallback to old model
]
for xgb_path in xgb_paths:
    if xgb_path.exists():
        try:
            models["xgboost"] = joblib.load(xgb_path)
            model_type = "NEW (IoT-IDAD + CICAPT)" if "new_datasets" in str(xgb_path) else "OLD (CICIDS2017)"
            logger.info(f"Loaded XGBoost model from {xgb_path} [{model_type}]")
            break
        except Exception as e:
            logger.warning(f"Failed to load XGBoost model from {xgb_path}: {e}")

# Try Isolation Forest - prefer new models
if_paths = [
    MODELS_DIR / "isolation_forest_new_datasets.pkl",  # New model
    MODELS_DIR / "isolation_forest.pkl",  # Fallback to old model
]
for if_path in if_paths:
    if if_path.exists():
        try:
            models["isolation_forest"] = joblib.load(if_path)
            model_type = "NEW (IoT-IDAD + CICAPT)" if "new_datasets" in str(if_path) else "OLD (CICIDS2017)"
            logger.info(f"Loaded Isolation Forest model from {if_path} [{model_type}]")
            break
        except Exception as e:
            logger.warning(f"Failed to load Isolation Forest model from {if_path}: {e}")

# Scaler - prefer new scaler
scaler = None
scaler_paths = [
    MODELS_DIR / "scaler_new_datasets.pkl",  # New scaler
    MODELS_DIR / "scaler.pkl",  # Fallback to old scaler
]
for scaler_path in scaler_paths:
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            scaler_type = "NEW" if "new_datasets" in str(scaler_path) else "OLD"
            logger.info(f"Loaded scaler from {scaler_path} [{scaler_type}]")
            break
        except Exception as e:
            logger.warning(f"Failed to load scaler from {scaler_path}: {e}")

# Feature names (used as selected features) - prefer new feature names
selected_features = []
feat_paths = [
    MODELS_DIR / "feature_names_new_datasets.pkl",  # New features
    MODELS_DIR / "selected_features.pkl",  # Fallback
    MODELS_DIR / "feature_names.pkl",  # Fallback
]
for p in feat_paths:
    if p.exists():
        try:
            selected_features = list(joblib.load(p))
            feat_type = "NEW" if "new_datasets" in str(p) else "OLD"
            logger.info(f"Loaded feature list from {p} [{feat_type}]")
            break
        except Exception as e:
            logger.warning(f"Failed to load feature list from {p}: {e}")

# Label encoder (optional)
label_encoder = None
le_path = MODELS_DIR / "label_encoder.pkl"
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
    "total_requests": 0,
    "threats_detected": 0,
    "threat_history": [],
    "current_threat_level": "Low",
}


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/test-data/sample", methods=["GET"])
@require_api_key(["read"])
def get_test_data_sample():
    """Get a sample test data for frontend testing"""
    import json
    from pathlib import Path
    
    test_data_path = PROJECT_ROOT / "test_data" / "test_data_new_models.json"
    
    if test_data_path.exists():
        try:
            with open(test_data_path, 'r') as f:
                all_samples = json.load(f)
            
            if all_samples:
                # Return a random sample
                import random
                sample = random.choice(all_samples)
                return jsonify({
                    "success": True,
                    "sample": sample,
                    "total_samples": len(all_samples)
                })
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
    
    # Fallback: Generate a simple test sample with ALL required features
    # Use selected_features to create a sample with all required features
    sample_features = {}
    if selected_features:
        for feat in selected_features:  # ALL features, not just first 50
            if "Port" in feat:
                sample_features[feat] = 80
            elif "Duration" in feat:
                sample_features[feat] = 1000
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                sample_features[feat] = 10000
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                sample_features[feat] = 0
            elif "Bytes/s" in feat or "Bytes per" in feat or "Bytes" in feat:
                sample_features[feat] = 15000000
            elif "Packets/s" in feat or "Packets per" in feat:
                sample_features[feat] = 10000
            elif "IAT" in feat or "Inter Arrival" in feat:
                sample_features[feat] = 0.05
            elif "Active" in feat:
                sample_features[feat] = 1000
            elif "Idle" in feat:
                sample_features[feat] = 0
            elif "Flag" in feat or "Flags" in feat:
                sample_features[feat] = 0
            elif "Length" in feat:
                sample_features[feat] = 1500
            elif "Mean" in feat or "Avg" in feat or "Average" in feat:
                sample_features[feat] = 1000
            elif "Std" in feat or "Variance" in feat:
                sample_features[feat] = 100
            elif "Max" in feat:
                sample_features[feat] = 1500
            elif "Min" in feat:
                sample_features[feat] = 0
            else:
                # Default value for any other features
                sample_features[feat] = 100
    
    return jsonify({
        "success": True,
        "sample": {
            "features": sample_features,
            "source_ip": "192.168.1.100",
            "attack_type": "test_sample"
        },
        "total_samples": 1,
        "note": "Generated sample (test data file not available)"
    })


@app.route("/predict", methods=["POST"])
@require_api_key(["read", "write"])
def predict():
    """Main prediction endpoint with comprehensive error handling"""
    start_time = time.time()

    try:
        # Validate request
        if not models:
            return jsonify({"error": "No models loaded"}), 503

        data = request.json or {}

        # Comprehensive data validation
        validated_data = validate_request(data)

        # Check for anomalous input
        is_anomalous, anomalies = detect_anomalies(data.get("features", {}))
        if is_anomalous:
            log_security(
                {
                    "severity": "HIGH",
                    "event": "Anomalous Input Detected",
                    "description": f"Potentially malicious input detected: {anomalies}",
                    "source": data.get("source_ip", request.remote_addr),
                    "details": {"anomalies": anomalies},
                }
            )

        # Prepare features for model input
        X = prepare_features(validated_data["model_features"])
    except Exception as e:
        logger.error(f"Request validation error: {str(e)}")
        return jsonify({"error": f"Validation error: {str(e)}"}), 400

    try:
        # Get predictions from all models with error handling
        predictions = {}
        threat_scores = []

        for model_name, model in models.items():
            model_start = time.time()
            prob_value = None  # Initialize prob_value
            try:
                if hasattr(model, "predict_proba"):
                    prob = safe_model_prediction(model, X, model_name)
                    prob_value = float(prob[0, 1]) if prob.ndim > 1 else float(prob[0])
                    predictions[model_name] = prob_value
                    threat_scores.append(prob_value)
                else:
                    # For anomaly detection (Isolation Forest)
                    # Isolation Forest returns -1 for anomalies, 1 for normal
                    pred = safe_model_prediction(model, X, model_name)
                    # Convert: -1 (anomaly) -> 1.0 (high threat), 1 (normal) -> 0.0 (low threat)
                    if isinstance(pred, np.ndarray):
                        score = 1.0 if int(pred[0]) == -1 else 0.0
                    else:
                        score = 1.0 if int(pred) == -1 else 0.0
                    predictions[model_name] = score
                    threat_scores.append(score)
                    prob_value = score  # For consistency with other models

                # Update model performance metrics
                model_time = (time.time() - model_start) * 1000
                if model_name in model_performance["models"]:
                    stats = model_performance["models"][model_name]
                    stats["predictions"] += 1
                    # Running average for confidence
                    n = stats["predictions"]
                    if prob_value is not None:
                        stats["avg_confidence"] = ((stats["avg_confidence"] * (n-1)) + prob_value) / n
                    # Running average for time
                    stats["avg_time_ms"] = ((stats["avg_time_ms"] * (n-1)) + model_time) / n
                    stats["status"] = "healthy"
                    stats["contribution_weight"] = 100.0 / len(models)  # Equal weight for now

            except Exception as e:
                handle_model_error(e, model_name)
                logger.warning(f"Model {model_name} prediction failed: {e}")
                if model_name in model_performance["models"]:
                    model_performance["models"][model_name]["status"] = "failed"

        if not threat_scores:
            return jsonify({"error": "No predictions available"}), 500

        # Calculate average threat score
        avg_threat_score = float(np.mean(threat_scores))

        # Determine threat level
        if avg_threat_score >= 0.8:
            threat_level = "Critical"
        elif avg_threat_score >= 0.6:
            threat_level = "High"
        elif avg_threat_score >= 0.4:
            threat_level = "Medium"
        elif avg_threat_score >= 0.2:
            threat_level = "Low"
        else:
            threat_level = "None"

        # Update statistics
        threat_stats["total_requests"] += 1
        model_performance["total_predictions"] += 1
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Log threat detection
        threat_data = {
            "threat_level": threat_level,
            "threat_score": avg_threat_score,
            "source_ip": data.get("source_ip", "Unknown"),
            "attack_type": data.get("attack_type", "Unknown"),
            "model_predictions": predictions,
            "processing_time": processing_time,
            "user_agent": request.headers.get("User-Agent", "Unknown"),
        }
        log_threat(threat_data)

        # Store in database
        try:
            store_threat(threat_data)
        except Exception as e:
            logger.warning(f"Failed to store threat in database: {e}")

        if threat_level in ["Critical", "High", "Medium", "Low"]:
            threat_stats["threats_detected"] += 1

            # Create alert
            alert = {
                "timestamp": datetime.now().isoformat(),
                "threat_level": threat_level,
                "threat_score": avg_threat_score,
                "source_ip": data.get("source_ip", "Unknown"),
                "attack_type": data.get("attack_type", "Unknown"),
            }

            # Add to alert queue and emit
            alert_queue.put(alert)
            socketio.emit("new_alert", alert)

            # Store alert in database
            try:
                store_alert(alert)
            except Exception as e:
                logger.warning(f"Failed to store alert in database: {e}")

            # Log security event
            log_security(
                {
                    "severity": "HIGH" if threat_level == "High" else "CRITICAL",
                    "event": f"Threat Detected: {threat_level}",
                    "description": f"Threat level {threat_level} detected with score {avg_threat_score:.3f}",
                    "source": data.get("source_ip", "Unknown"),
                    "details": alert,
                }
            )

        # Add to history
        threat_stats["threat_history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "threat_score": avg_threat_score,
                "threat_level": threat_level,
            }
        )

        # Keep only last 100 entries
        if len(threat_stats["threat_history"]) > 100:
            threat_stats["threat_history"] = threat_stats["threat_history"][-100:]

        threat_stats["current_threat_level"] = threat_level

        # Log API request
        api_usage_data = {
            "endpoint": "/predict",
            "method": "POST",
            "response_time": processing_time,
            "status_code": 200,
            "client_ip": request.remote_addr,
            "user_agent": request.headers.get("User-Agent", "Unknown"),
            "api_key": "Unknown",
        }
        log_api_request(api_usage_data)

        # Store API usage in database
        try:
            store_api_usage(api_usage_data)
        except Exception as e:
            logger.warning(f"Failed to store API usage in database: {e}")

        response = {
            "threat_detected": threat_level in ["Critical", "High", "Medium"],
            "threat_level": threat_level,
            "threat_score": avg_threat_score,
            "model_predictions": predictions,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
@require_api_key(["read"])
def get_stats():
    """Get system statistics"""
    return jsonify(threat_stats)


@app.route("/alerts", methods=["GET"])
@require_api_key(["read"])
def get_alerts():
    """Get recent alerts"""
    alerts = []
    while not alert_queue.empty() and len(alerts) < 50:
        alerts.append(alert_queue.get())

    return jsonify({"alerts": alerts})


@app.route("/system/info", methods=["GET"])
@require_api_key(["read", "admin"])
def system_info():
    """Get system information with health monitoring"""
    import psutil

    try:
        health_data = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
            "active_connections": len(models),
            "queue_size": alert_queue.qsize(),
        }

        # Log system health
        log_health(health_data)

        # Store system metrics in database
        try:
            store_system_metrics(health_data)
        except Exception as e:
            logger.warning(f"Failed to store system metrics in database: {e}")

        info = {
            "models_loaded": list(models.keys()),
            "cpu_percent": health_data["cpu_percent"],
            "memory_percent": health_data["memory_percent"],
            "disk_usage": health_data["disk_usage"],
            "total_predictions": threat_stats["total_requests"],
            "threats_detected": threat_stats["threats_detected"],
            "detection_rate": (
                threat_stats["threats_detected"]
                / max(threat_stats["total_requests"], 1)
            )
            * 100,
            "alert_queue_size": health_data["queue_size"],
        }

        return jsonify(info)

    except Exception as e:
        handle_api_error(e, "/system/info")
        raise


# WebSocket events
@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    emit("connected", {"data": "Connected to threat detection system"})


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")


@app.route("/errors", methods=["GET"])
@require_api_key(["admin"])
def get_errors():
    """Get error statistics and recent errors"""
    try:
        error_stats = get_error_summary()
        return jsonify(error_stats)
    except Exception as e:
        handle_api_error(e, "/errors")
        raise


@app.route("/admin/keys", methods=["GET"])
@require_api_key(["admin"])
def list_api_keys():
    """List all API keys (admin only)"""
    try:
        keys_info = []
        for key, config in api_key_manager.api_keys.items():
            keys_info.append(
                {
                    "key": key[:8] + "..." + key[-4:],  # Mask the key
                    "name": config["name"],
                    "permissions": config["permissions"],
                    "rate_limit": config["rate_limit"],
                    "expires": config["expires"],
                    "created": config["created"],
                }
            )
        return jsonify({"api_keys": keys_info})
    except Exception as e:
        handle_api_error(e, "/admin/keys")
        raise


@app.route("/admin/keys", methods=["POST"])
@require_api_key(["admin"])
def create_new_api_key():
    """Create a new API key (admin only)"""
    try:
        data = request.json or {}
        name = data.get("name", "New API Key")
        permissions = data.get("permissions", ["read"])
        rate_limit = data.get("rate_limit", 100)

        if not isinstance(permissions, list):
            return jsonify({"error": "Permissions must be a list"}), 400

        new_key = create_api_key(name, permissions, rate_limit)

        return jsonify(
            {
                "api_key": new_key,
                "name": name,
                "permissions": permissions,
                "rate_limit": rate_limit,
                "message": "API key created successfully",
            }
        )
    except Exception as e:
        handle_api_error(e, "/admin/keys")
        raise


@app.route("/admin/keys/<key>", methods=["DELETE"])
@require_api_key(["admin"])
def revoke_api_key_endpoint(key):
    """Revoke an API key (admin only)"""
    try:
        success = revoke_api_key(key)
        if success:
            return jsonify({"message": "API key revoked successfully"})
        else:
            return jsonify({"error": "API key not found"}), 404
    except Exception as e:
        handle_api_error(e, f"/admin/keys/{key}")
        raise


@app.route("/admin/security", methods=["GET"])
@require_api_key(["admin"])
def get_security_info():
    """Get security information (admin only)"""
    try:
        from utils.auth import security_middleware

        security_stats = security_middleware.get_security_stats()

        return jsonify(
            {
                "security_stats": security_stats,
                "blocked_ips": list(security_middleware.blocked_ips),
                "suspicious_ips": list(security_middleware.suspicious_ips.keys()),
            }
        )
    except Exception as e:
        handle_api_error(e, "/admin/security")
        raise


@app.route("/database/threats", methods=["GET"])
@require_api_key(["read", "admin"])
def get_database_threats():
    """Get threats from database with filtering"""
    try:
        limit = int(request.args.get("limit", 100))
        threat_level = request.args.get("threat_level")
        days = int(request.args.get("days", 7))

        if days:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
        else:
            start_date = None

        threats = get_threats(
            limit=limit, threat_level=threat_level, start_date=start_date
        )

        return jsonify(
            {
                "threats": threats,
                "count": len(threats),
                "filters": {"limit": limit, "threat_level": threat_level, "days": days},
            }
        )
    except Exception as e:
        handle_api_error(e, "/database/threats")
        raise


@app.route("/database/alerts", methods=["GET"])
@require_api_key(["read", "admin"])
def get_database_alerts():
    """Get alerts from database"""
    try:
        limit = int(request.args.get("limit", 100))
        acknowledged = request.args.get("acknowledged")

        if acknowledged is not None:
            acknowledged = acknowledged.lower() == "true"

        alerts = get_alerts(limit=limit, acknowledged=acknowledged)

        return jsonify(
            {
                "alerts": alerts,
                "count": len(alerts),
                "filters": {"limit": limit, "acknowledged": acknowledged},
            }
        )
    except Exception as e:
        handle_api_error(e, "/database/alerts")
        raise


@app.route("/database/statistics", methods=["GET"])
@require_api_key(["read", "admin"])
def get_database_statistics():
    """Get threat statistics from database"""
    try:
        days = int(request.args.get("days", 7))
        stats = get_threat_statistics(days=days)

        # Add database stats
        db_stats = threat_db.get_database_stats()
        stats["database"] = db_stats

        return jsonify(stats)
    except Exception as e:
        handle_api_error(e, "/database/statistics")
        raise


@app.route("/database/alerts/<int:alert_id>/acknowledge", methods=["POST"])
@require_api_key(["admin"])
def acknowledge_database_alert(alert_id):
    """Acknowledge an alert in the database"""
    try:
        user = get_current_user()
        acknowledged_by = user["name"] if user else "Unknown"

        success = threat_db.acknowledge_alert(alert_id, acknowledged_by)

        if success:
            return jsonify({"message": f"Alert {alert_id} acknowledged successfully"})
        else:
            return jsonify({"error": "Alert not found"}), 404
    except Exception as e:
        handle_api_error(e, f"/database/alerts/{alert_id}/acknowledge")
        raise


@app.route("/database/cleanup", methods=["POST"])
@require_api_key(["admin"])
def cleanup_database():
    """Clean up old database records"""
    try:
        days = int(request.json.get("days", 30)) if request.json else 30
        threat_db.cleanup_old_data(days=days)

        return jsonify(
            {
                "message": f"Database cleanup completed for records older than {days} days"
            }
        )
    except Exception as e:
        handle_api_error(e, "/database/cleanup")
        raise


# Model performance tracking
model_performance = {
    "total_predictions": 0,
    "healthy_models": 0,
    "total_models": len(models),
    "models": {}
}

# Initialize model stats
for model_name in models.keys():
    model_performance["models"][model_name] = {
        "predictions": 0,
        "avg_confidence": 0.0,
        "avg_time_ms": 0.0,
        "contribution_weight": 0.0,
        "status": "ready",
        "available": True
    }


@app.route("/models/performance", methods=["GET"])
@require_api_key(["read"])
def get_model_performance():
    """Get model performance metrics"""
    try:
        # Update healthy models count
        healthy_count = sum(1 for m in model_performance["models"].values()
                          if m["status"] == "healthy" or m["status"] == "ready")
        model_performance["healthy_models"] = healthy_count
        model_performance["total_models"] = len(models)

        return jsonify(model_performance)
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
@require_api_key(["read"])
def explain_prediction():
    """Explain prediction with feature importance"""
    try:
        data = request.get_json()
        features = data.get("features", {})

        if not features:
            return jsonify({"error": "No features provided"}), 400

        # Get feature names and values
        feature_names = list(features.keys())
        feature_values = list(features.values())

        # Calculate simple feature importance (using feature magnitude)
        total = sum(abs(v) for v in feature_values if isinstance(v, (int, float)))
        if total == 0:
            weights = [1.0 / len(feature_values)] * len(feature_values)
        else:
            weights = [abs(v) / total if isinstance(v, (int, float)) else 0.0
                      for v in feature_values]

        # Sort features by importance
        feature_importance = sorted(zip(feature_names, weights),
                                   key=lambda x: x[1], reverse=True)

        # Generate explanation text
        top_features = feature_importance[:3]
        explanation = f"Threat Level: Low\n"
        explanation += "Key indicators:\n"
        for feat, score in top_features:
            importance = "High" if score > 0.15 else "Medium" if score > 0.10 else "Low"
            explanation += f"- {feat}: {importance} importance ({score*100:.1f}%)\n"

        return jsonify({
            "success": True,
            "explanation": explanation.strip(),
            "threat_level": "Low",
            "top_features": feature_importance[:5],
            "visualization_data": {
                "features": feature_names,
                "weights": weights
            }
        })
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001, allow_unsafe_werkzeug=True)
