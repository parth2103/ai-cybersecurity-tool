#!/usr/bin/env python3
"""
Data validation and sanitization system for AI Cybersecurity Tool
Provides comprehensive data validation, cleaning, and security checks
"""

import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import joblib

from .error_handler import DataValidationError, ErrorSeverity
from .logger import get_logger

logger = get_logger("data_validator")


class DataValidator:
    """Comprehensive data validation and sanitization system"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.feature_names = self._load_feature_names()
        self.scaler = self._load_scaler()
        self.validation_rules = self._setup_validation_rules()

    def _load_feature_names(self) -> List[str]:
        """Load expected feature names"""
        try:
            feat_paths = [
                self.models_dir / "selected_features.pkl",
                self.models_dir / "feature_names.pkl",
            ]
            for path in feat_paths:
                if path.exists():
                    features = joblib.load(path)
                    logger.info(f"Loaded {len(features)} feature names from {path}")
                    return list(features)
            logger.warning("No feature names found, using empty list")
            return []
        except Exception as e:
            logger.error(f"Failed to load feature names: {e}")
            return []

    def _load_scaler(self):
        """Load the scaler for validation"""
        try:
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler for validation")
                return scaler
            return None
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return None

    def _setup_validation_rules(self) -> Dict[str, Dict]:
        """Setup validation rules for different data types"""
        return {
            "ip_address": {
                "pattern": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
                "description": "Valid IPv4 address",
            },
            "port": {
                "min": 1,
                "max": 65535,
                "description": "Valid port number (1-65535)",
            },
            "protocol": {
                "allowed": ["TCP", "UDP", "ICMP", "tcp", "udp", "icmp"],
                "description": "Valid network protocol",
            },
            "numeric": {
                "min": -1e6,
                "max": 1e6,
                "description": "Numeric value within reasonable range",
            },
            "positive_numeric": {
                "min": 0,
                "max": 1e6,
                "description": "Positive numeric value",
            },
        }

    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format"""
        if not isinstance(ip, str):
            return False

        pattern = self.validation_rules["ip_address"]["pattern"]
        return bool(re.match(pattern, ip))

    def validate_port(self, port: Union[int, str]) -> bool:
        """Validate port number"""
        try:
            port_num = int(port)
            return 1 <= port_num <= 65535
        except (ValueError, TypeError):
            return False

    def validate_protocol(self, protocol: str) -> bool:
        """Validate network protocol"""
        if not isinstance(protocol, str):
            return False

        allowed = self.validation_rules["protocol"]["allowed"]
        return protocol.upper() in [p.upper() for p in allowed]

    def validate_numeric_range(
        self, value: Any, min_val: float = None, max_val: float = None
    ) -> bool:
        """Validate numeric value is within range"""
        try:
            num_val = float(value)
            if min_val is not None and num_val < min_val:
                return False
            if max_val is not None and num_val > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False

    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return str(value)

        # Remove null bytes and control characters
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", value)

        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized.strip()

    def sanitize_numeric(self, value: Any) -> float:
        """Sanitize numeric input"""
        try:
            if pd.isna(value) or value is None:
                return 0.0

            num_val = float(value)

            # Handle infinite values
            if np.isinf(num_val):
                return 1e6 if num_val > 0 else -1e6

            # Handle NaN
            if np.isnan(num_val):
                return 0.0

            return num_val
        except (ValueError, TypeError):
            return 0.0

    def validate_network_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate network-specific features"""
        validated_features = {}
        errors = []

        # Validate IP addresses
        ip_fields = ["source_ip", "destination_ip", "src_ip", "dst_ip"]
        for field in ip_fields:
            if field in features:
                value = features[field]
                if value and not self.validate_ip_address(value):
                    errors.append(f"Invalid IP address in {field}: {value}")
                validated_features[field] = self.sanitize_string(value, 15)

        # Validate ports
        port_fields = ["source_port", "destination_port", "src_port", "dst_port"]
        for field in port_fields:
            if field in features:
                value = features[field]
                if value and not self.validate_port(value):
                    errors.append(f"Invalid port in {field}: {value}")
                validated_features[field] = int(self.sanitize_numeric(value))

        # Validate protocol
        if "protocol" in features:
            value = features["protocol"]
            if value and not self.validate_protocol(value):
                errors.append(f"Invalid protocol: {value}")
            validated_features["protocol"] = self.sanitize_string(value, 10)

        if errors:
            raise DataValidationError(
                f"Network validation errors: {'; '.join(errors)}",
                severity=ErrorSeverity.MEDIUM,
            )

        return validated_features

    def validate_model_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate features for model prediction"""
        validated_features = {}
        errors = []

        # Check required features
        if self.feature_names:
            missing_features = [f for f in self.feature_names if f not in features]
            if missing_features:
                errors.append(f"Missing required features: {missing_features}")

        # Validate and sanitize all features
        for key, value in features.items():
            try:
                # Sanitize key
                clean_key = self.sanitize_string(key, 100)

                # Sanitize value based on type
                if isinstance(value, str):
                    clean_value = self.sanitize_string(value, 1000)
                elif isinstance(value, (int, float)):
                    clean_value = self.sanitize_numeric(value)
                else:
                    clean_value = self.sanitize_numeric(value)

                validated_features[clean_key] = clean_value

            except Exception as e:
                errors.append(f"Error validating feature {key}: {str(e)}")

        if errors:
            raise DataValidationError(
                f"Feature validation errors: {'; '.join(errors)}",
                severity=ErrorSeverity.HIGH,
            )

        return validated_features

    def validate_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete API request"""
        errors = []
        validated_data = {}

        # Validate request structure
        if not isinstance(request_data, dict):
            raise DataValidationError(
                "Request data must be a dictionary", severity=ErrorSeverity.HIGH
            )

        # Validate features
        if "features" not in request_data:
            raise DataValidationError(
                "Missing 'features' field", severity=ErrorSeverity.HIGH
            )

        features = request_data["features"]
        if not isinstance(features, dict):
            raise DataValidationError(
                "Features must be a dictionary", severity=ErrorSeverity.HIGH
            )

        # Validate network features
        try:
            validated_data["network_features"] = self.validate_network_features(
                features
            )
        except DataValidationError as e:
            errors.append(str(e))

        # Validate model features
        try:
            validated_data["model_features"] = self.validate_model_features(features)
        except DataValidationError as e:
            errors.append(str(e))

        # Validate optional fields
        optional_fields = ["source_ip", "attack_type", "timestamp"]
        for field in optional_fields:
            if field in request_data:
                value = request_data[field]
                if field in ["source_ip"]:
                    if value and not self.validate_ip_address(value):
                        errors.append(f"Invalid {field}: {value}")
                    validated_data[field] = self.sanitize_string(value, 15)
                else:
                    validated_data[field] = self.sanitize_string(str(value), 100)

        if errors:
            raise DataValidationError(
                f"API request validation errors: {'; '.join(errors)}",
                severity=ErrorSeverity.HIGH,
            )

        return validated_data

    def prepare_model_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare validated features for model input"""
        try:
            # Ensure all required features are present
            if self.feature_names:
                for feature in self.feature_names:
                    if feature not in features:
                        features[feature] = 0.0

            # Create DataFrame with proper feature order
            if self.feature_names:
                df = pd.DataFrame([features])[self.feature_names]
            else:
                df = pd.DataFrame([features])

            # Convert to numpy array
            X = df.values

            # Apply scaling if available
            if self.scaler is not None:
                X = self.scaler.transform(df)

            return X

        except Exception as e:
            raise DataValidationError(
                f"Failed to prepare model input: {str(e)}", severity=ErrorSeverity.HIGH
            )

    def detect_anomalous_input(
        self, features: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Detect potentially anomalous or malicious input"""
        anomalies = []

        # Check for suspicious values
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Check for extreme values
                if abs(value) > 1e6:
                    anomalies.append(f"Extreme value in {key}: {value}")

                # Check for NaN or infinite values
                if np.isnan(value) or np.isinf(value):
                    anomalies.append(f"Invalid numeric value in {key}: {value}")

            elif isinstance(value, str):
                # Check for suspicious strings
                if len(value) > 10000:  # Very long strings
                    anomalies.append(f"Unusually long string in {key}")

                # Check for potential injection patterns
                suspicious_patterns = [
                    r"<script",
                    r"javascript:",
                    r"data:",
                    r"vbscript:",
                    r"<iframe",
                    r"<object",
                    r"<embed",
                    r"<link",
                    r"SELECT\s+.*\s+FROM",
                    r"INSERT\s+INTO",
                    r"UPDATE\s+.*\s+SET",
                    r"DELETE\s+FROM",
                    r"DROP\s+TABLE",
                    r"UNION\s+SELECT",
                ]

                for pattern in suspicious_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        anomalies.append(f"Suspicious pattern in {key}: {pattern}")

        # Check for too many features (potential DoS)
        if len(features) > 1000:
            anomalies.append(f"Too many features: {len(features)}")

        return len(anomalies) > 0, anomalies

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation system summary"""
        return {
            "feature_count": len(self.feature_names),
            "has_scaler": self.scaler is not None,
            "validation_rules": len(self.validation_rules),
            "models_dir": str(self.models_dir),
        }


# Global validator instance
validator = DataValidator()


def validate_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate API request data"""
    return validator.validate_api_request(request_data)


def prepare_features(features: Dict[str, Any]) -> np.ndarray:
    """Prepare features for model input"""
    return validator.prepare_model_input(features)


def detect_anomalies(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Detect anomalous input"""
    return validator.detect_anomalous_input(features)


def get_validator_summary() -> Dict[str, Any]:
    """Get validator summary"""
    return validator.get_validation_summary()
