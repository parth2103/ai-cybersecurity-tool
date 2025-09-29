#!/usr/bin/env python3
"""
Comprehensive logging system for AI Cybersecurity Tool
Provides structured logging with different levels and handlers
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
import json
import traceback
from typing import Optional, Dict, Any


class SecurityLogger:
    """Enhanced logger for cybersecurity tool with structured logging"""

    def __init__(self, name: str = "ai_cybersecurity", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Setup different log handlers"""

        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (DEBUG and above)
        file_handler = logging.FileHandler(self.log_dir / f"{self.name}.log", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Error handler (ERROR and above)
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_errors.log", mode="a"
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n%(pathname)s:%(lineno)d\n",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)

        # Rotating file handler for main log
        rotating_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_rotating.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        rotating_handler.setLevel(logging.INFO)
        rotating_handler.setFormatter(file_formatter)
        self.logger.addHandler(rotating_handler)

    def log_threat_detection(self, threat_data: Dict[str, Any]):
        """Log threat detection events with structured data"""
        threat_log = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "threat_detection",
            "threat_level": threat_data.get("threat_level", "Unknown"),
            "threat_score": threat_data.get("threat_score", 0),
            "source_ip": threat_data.get("source_ip", "Unknown"),
            "attack_type": threat_data.get("attack_type", "Unknown"),
            "model_predictions": threat_data.get("model_predictions", {}),
            "processing_time": threat_data.get("processing_time", 0),
        }

        self.logger.info(f"THREAT_DETECTED: {json.dumps(threat_log)}")

    def log_api_request(self, request_data: Dict[str, Any]):
        """Log API requests with performance metrics"""
        api_log = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "api_request",
            "endpoint": request_data.get("endpoint", "Unknown"),
            "method": request_data.get("method", "Unknown"),
            "response_time": request_data.get("response_time", 0),
            "status_code": request_data.get("status_code", 0),
            "client_ip": request_data.get("client_ip", "Unknown"),
            "user_agent": request_data.get("user_agent", "Unknown"),
        }

        self.logger.info(f"API_REQUEST: {json.dumps(api_log)}")

    def log_model_performance(self, performance_data: Dict[str, Any]):
        """Log model performance metrics"""
        perf_log = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "model_performance",
            "model_name": performance_data.get("model_name", "Unknown"),
            "accuracy": performance_data.get("accuracy", 0),
            "precision": performance_data.get("precision", 0),
            "recall": performance_data.get("recall", 0),
            "f1_score": performance_data.get("f1_score", 0),
            "inference_time": performance_data.get("inference_time", 0),
            "memory_usage": performance_data.get("memory_usage", 0),
        }

        self.logger.info(f"MODEL_PERFORMANCE: {json.dumps(perf_log)}")

    def log_system_health(self, health_data: Dict[str, Any]):
        """Log system health metrics"""
        health_log = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "system_health",
            "cpu_percent": health_data.get("cpu_percent", 0),
            "memory_percent": health_data.get("memory_percent", 0),
            "disk_usage": health_data.get("disk_usage", 0),
            "active_connections": health_data.get("active_connections", 0),
            "queue_size": health_data.get("queue_size", 0),
        }

        self.logger.info(f"SYSTEM_HEALTH: {json.dumps(health_log)}")

    def log_security_event(self, event_data: Dict[str, Any]):
        """Log security-related events"""
        security_log = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "security_event",
            "severity": event_data.get("severity", "INFO"),
            "event": event_data.get("event", "Unknown"),
            "description": event_data.get("description", ""),
            "source": event_data.get("source", "Unknown"),
            "details": event_data.get("details", {}),
        }

        if security_log["severity"] == "CRITICAL":
            self.logger.critical(f"SECURITY_EVENT: {json.dumps(security_log)}")
        elif security_log["severity"] == "HIGH":
            self.logger.error(f"SECURITY_EVENT: {json.dumps(security_log)}")
        else:
            self.logger.warning(f"SECURITY_EVENT: {json.dumps(security_log)}")

    def log_exception(self, exception: Exception, context: str = ""):
        """Log exceptions with full traceback"""
        exception_log = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "exception",
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "context": context,
            "traceback": traceback.format_exc(),
        }

        self.logger.error(f"EXCEPTION: {json.dumps(exception_log)}")

    def debug(self, message: str):
        """Debug level logging"""
        self.logger.debug(message)

    def info(self, message: str):
        """Info level logging"""
        self.logger.info(message)

    def warning(self, message: str):
        """Warning level logging"""
        self.logger.warning(message)

    def error(self, message: str):
        """Error level logging"""
        self.logger.error(message)

    def critical(self, message: str):
        """Critical level logging"""
        self.logger.critical(message)


# Global logger instance
logger = SecurityLogger()


def get_logger(name: Optional[str] = None) -> SecurityLogger:
    """Get logger instance"""
    if name:
        return SecurityLogger(name)
    return logger


# Convenience functions
def log_threat(threat_data: Dict[str, Any]):
    """Log threat detection event"""
    logger.log_threat_detection(threat_data)


def log_api_request(request_data: Dict[str, Any]):
    """Log API request"""
    logger.log_api_request(request_data)


def log_performance(performance_data: Dict[str, Any]):
    """Log model performance"""
    logger.log_model_performance(performance_data)


def log_health(health_data: Dict[str, Any]):
    """Log system health"""
    logger.log_system_health(health_data)


def log_security(event_data: Dict[str, Any]):
    """Log security event"""
    logger.log_security_event(event_data)


def log_exception(exception: Exception, context: str = ""):
    """Log exception"""
    logger.log_exception(exception, context)
