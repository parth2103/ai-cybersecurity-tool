#!/usr/bin/env python3
"""
Comprehensive error handling system for AI Cybersecurity Tool
Provides centralized error handling, validation, and recovery mechanisms
"""

import functools
import traceback
import time
from typing import Any, Callable, Dict, Optional, Union
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path

from .logger import get_logger

logger = get_logger("error_handler")


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""

    DATA_VALIDATION = "data_validation"
    MODEL_ERROR = "model_error"
    API_ERROR = "api_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    SECURITY_ERROR = "security_error"


class SecurityError(Exception):
    """Custom security-related error"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        category: ErrorCategory = ErrorCategory.SECURITY_ERROR,
    ):
        self.message = message
        self.severity = severity
        self.category = category
        super().__init__(self.message)


class ModelError(Exception):
    """Custom model-related error"""

    def __init__(
        self,
        message: str,
        model_name: str = "unknown",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        self.message = message
        self.model_name = model_name
        self.severity = severity
        self.category = ErrorCategory.MODEL_ERROR
        super().__init__(self.message)


class DataValidationError(Exception):
    """Custom data validation error"""

    def __init__(
        self,
        message: str,
        field: str = "unknown",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        self.message = message
        self.field = field
        self.severity = severity
        self.category = ErrorCategory.DATA_VALIDATION
        super().__init__(self.message)


class ErrorHandler:
    """Centralized error handling system"""

    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000

    def handle_error(
        self,
        error: Exception,
        context: str = "",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
    ) -> Dict[str, Any]:
        """Handle and log errors with context"""

        error_info = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "severity": severity.value,
            "category": category.value,
            "traceback": traceback.format_exc(),
        }

        # Log the error
        logger.log_exception(error, context)

        # Update error counts
        error_key = f"{error_info['error_type']}_{category.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history :]

        # Log security event if critical
        if severity == ErrorSeverity.CRITICAL:
            logger.log_security(
                {
                    "severity": "CRITICAL",
                    "event": "Critical Error Occurred",
                    "description": f"{error_info['error_type']}: {error_info['error_message']}",
                    "source": context,
                    "details": error_info,
                }
            )

        return error_info

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else [],
        }

    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.error_counts.clear()


# Global error handler
error_handler = ErrorHandler()


def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
    reraise: bool = False,
    fallback_value: Any = None,
):
    """Decorator for automatic error handling"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = f"{func.__name__} with args={args}, kwargs={kwargs}"
                error_info = error_handler.handle_error(e, context, severity, category)

                if reraise:
                    raise

                # For Flask API endpoints, return proper error response
                if "request" in str(args) or "app" in str(args):
                    from flask import jsonify

                    return (
                        jsonify(
                            {
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "timestamp": time.time(),
                            }
                        ),
                        500,
                    )

                return fallback_value

        return wrapper

    return decorator


def validate_data(
    data: Any, data_type: str, required_fields: Optional[list] = None
) -> bool:
    """Validate input data"""
    try:
        if data is None:
            raise DataValidationError(
                f"{data_type} data is None", severity=ErrorSeverity.HIGH
            )

        if isinstance(data, dict):
            if required_fields:
                missing_fields = [
                    field for field in required_fields if field not in data
                ]
                if missing_fields:
                    raise DataValidationError(
                        f"Missing required fields: {missing_fields}",
                        field=",".join(missing_fields),
                        severity=ErrorSeverity.HIGH,
                    )

        elif isinstance(data, (list, np.ndarray)):
            if len(data) == 0:
                raise DataValidationError(
                    f"{data_type} data is empty", severity=ErrorSeverity.MEDIUM
                )

        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise DataValidationError(
                    f"{data_type} DataFrame is empty", severity=ErrorSeverity.MEDIUM
                )

        return True

    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(
            f"Data validation failed: {str(e)}", severity=ErrorSeverity.HIGH
        )


def validate_model_input(
    features: Union[dict, np.ndarray, pd.DataFrame],
    expected_features: Optional[list] = None,
) -> bool:
    """Validate model input features"""
    try:
        if isinstance(features, dict):
            if expected_features:
                missing_features = [f for f in expected_features if f not in features]
                if missing_features:
                    raise DataValidationError(
                        f"Missing model features: {missing_features}",
                        field=",".join(missing_features),
                        severity=ErrorSeverity.HIGH,
                    )

            # Check for None values
            none_features = [k for k, v in features.items() if v is None]
            if none_features:
                raise DataValidationError(
                    f"Features with None values: {none_features}",
                    field=",".join(none_features),
                    severity=ErrorSeverity.MEDIUM,
                )

        elif isinstance(features, (np.ndarray, pd.DataFrame)):
            if features.size == 0:
                raise DataValidationError(
                    "Model input is empty", severity=ErrorSeverity.HIGH
                )

            # Check for NaN values
            if isinstance(features, np.ndarray):
                if np.any(np.isnan(features)):
                    raise DataValidationError(
                        "Model input contains NaN values", severity=ErrorSeverity.MEDIUM
                    )
            elif isinstance(features, pd.DataFrame):
                if features.isnull().any().any():
                    raise DataValidationError(
                        "Model input contains NaN values", severity=ErrorSeverity.MEDIUM
                    )

        return True

    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(
            f"Model input validation failed: {str(e)}", severity=ErrorSeverity.HIGH
        )


def safe_model_prediction(model, features, model_name: str = "unknown"):
    """Safely make model predictions with error handling"""
    try:
        validate_model_input(features)

        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(features)
            if predictions is None or predictions.size == 0:
                raise ModelError(
                    f"Model {model_name} returned empty predictions", model_name
                )
            return predictions
        elif hasattr(model, "predict"):
            predictions = model.predict(features)
            if predictions is None or predictions.size == 0:
                raise ModelError(
                    f"Model {model_name} returned empty predictions", model_name
                )
            return predictions
        else:
            raise ModelError(
                f"Model {model_name} has no prediction methods", model_name
            )

    except ModelError:
        raise
    except Exception as e:
        raise ModelError(f"Model prediction failed: {str(e)}", model_name)


def safe_file_operation(
    operation: Callable,
    file_path: Union[str, Path],
    operation_name: str = "file operation",
) -> Any:
    """Safely perform file operations with error handling"""
    try:
        file_path = Path(file_path)

        if operation_name in ["read", "load"] and not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if operation_name in ["write", "save"]:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        return operation(file_path)

    except Exception as e:
        error_handler.handle_error(
            e,
            f"{operation_name} on {file_path}",
            ErrorSeverity.MEDIUM,
            ErrorCategory.SYSTEM_ERROR,
        )
        raise


def retry_on_failure(
    max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0
):
    """Decorator for retrying operations on failure"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {wait_time:.2f}s: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        error_handler.handle_error(
                            e,
                            f"Retry failed for {func.__name__}",
                            ErrorSeverity.HIGH,
                            ErrorCategory.SYSTEM_ERROR,
                        )

            raise last_exception

        return wrapper

    return decorator


def validate_api_request(request_data: dict, required_fields: list) -> bool:
    """Validate API request data"""
    try:
        if not isinstance(request_data, dict):
            raise DataValidationError(
                "Request data must be a dictionary", severity=ErrorSeverity.HIGH
            )

        missing_fields = [
            field for field in required_fields if field not in request_data
        ]
        if missing_fields:
            raise DataValidationError(
                f"Missing required fields: {missing_fields}",
                field=",".join(missing_fields),
                severity=ErrorSeverity.HIGH,
            )

        return True

    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(
            f"API request validation failed: {str(e)}", severity=ErrorSeverity.HIGH
        )


# Convenience functions
def handle_model_error(error: Exception, model_name: str = "unknown"):
    """Handle model-specific errors"""
    return error_handler.handle_error(
        error, f"Model: {model_name}", ErrorSeverity.MEDIUM, ErrorCategory.MODEL_ERROR
    )


def handle_api_error(error: Exception, endpoint: str = "unknown"):
    """Handle API-specific errors"""
    return error_handler.handle_error(
        error, f"API: {endpoint}", ErrorSeverity.MEDIUM, ErrorCategory.API_ERROR
    )


def handle_data_error(error: Exception, data_source: str = "unknown"):
    """Handle data-specific errors"""
    return error_handler.handle_error(
        error,
        f"Data: {data_source}",
        ErrorSeverity.MEDIUM,
        ErrorCategory.DATA_VALIDATION,
    )


def get_error_summary() -> Dict[str, Any]:
    """Get summary of all errors"""
    return error_handler.get_error_stats()
