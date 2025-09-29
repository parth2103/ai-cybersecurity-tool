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
models = {}

# Try RF baseline (support legacy name)
rf_paths = [
    MODELS_DIR / "baseline_model.pkl",
]
for p in rf_paths:
    if p.exists():
        try:
            models["rf"] = joblib.load(p)
            logger.info(f"Loaded RF model from {p}")
            break
        except Exception as e:
            logger.warning(f"Failed to load RF model from {p}: {e}")

# Try XGBoost
xgb_path = MODELS_DIR / "xgboost_model.pkl"
if xgb_path.exists():
    try:
        models["xgboost"] = joblib.load(xgb_path)
        logger.info(f"Loaded XGBoost model from {xgb_path}")
    except Exception as e:
        logger.warning(f"Failed to load XGBoost model: {e}")

# Try Isolation Forest
if_path = MODELS_DIR / "isolation_forest.pkl"
if if_path.exists():
    try:
        models["isolation_forest"] = joblib.load(if_path)
        logger.info(f"Loaded Isolation Forest model from {if_path}")
    except Exception as e:
        logger.warning(f"Failed to load Isolation Forest model: {e}")

# Scaler
scaler = None
scaler_path = MODELS_DIR / "scaler.pkl"
if scaler_path.exists():
    try:
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler.pkl")
    except Exception as e:
        logger.warning(f"Failed to load scaler.pkl: {e}")

# Feature names (used as selected features)
selected_features = []
feat_paths = [MODELS_DIR / "selected_features.pkl", MODELS_DIR / "feature_names.pkl"]
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
            try:
                if hasattr(model, "predict_proba"):
                    prob = safe_model_prediction(model, X, model_name)
                    prob_value = float(prob[0, 1]) if prob.ndim > 1 else float(prob[0])
                    predictions[model_name] = prob_value
                    threat_scores.append(prob_value)
                else:
                    # For anomaly detection
                    pred = safe_model_prediction(model, X, model_name)
                    score = 1.0 if int(pred[0]) == -1 else 0.0
                    predictions[model_name] = score
                    threat_scores.append(score)
            except Exception as e:
                handle_model_error(e, model_name)
                logger.warning(f"Model {model_name} prediction failed: {e}")

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

        if threat_level in ["Critical", "High"]:
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


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001, allow_unsafe_werkzeug=True)
