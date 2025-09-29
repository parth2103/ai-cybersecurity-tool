#!/usr/bin/env python3
"""
Authentication and authorization system for AI Cybersecurity Tool
Provides API key authentication, rate limiting, and security middleware
"""

import hashlib
import hmac
import time
import secrets
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from functools import wraps
import json

from flask import request, jsonify, g
from .logger import get_logger
from .error_handler import SecurityError, ErrorSeverity, ErrorCategory

logger = get_logger("auth")


class APIKeyManager:
    """Manages API keys and authentication"""

    def __init__(self):
        self.api_keys = {}
        self.key_usage = {}
        self.rate_limits = {}
        self._load_default_keys()

    def _load_default_keys(self):
        """Load default API keys for development"""
        # In production, these should be loaded from a secure database
        default_keys = {
            "dev-key-123": {
                "name": "Development Key",
                "permissions": ["read", "write", "admin"],
                "rate_limit": 1000,  # requests per hour
                "expires": None,  # Never expires
                "created": datetime.now().isoformat(),
            },
            "readonly-key-456": {
                "name": "Read-Only Key",
                "permissions": ["read"],
                "rate_limit": 100,  # requests per hour
                "expires": None,
                "created": datetime.now().isoformat(),
            },
            "admin-key-789": {
                "name": "Admin Key",
                "permissions": ["read", "write", "admin", "system"],
                "rate_limit": 10000,  # requests per hour
                "expires": None,
                "created": datetime.now().isoformat(),
            },
        }

        for key, config in default_keys.items():
            self.api_keys[key] = config
            self.key_usage[key] = []
            self.rate_limits[key] = config["rate_limit"]

    def generate_api_key(
        self,
        name: str,
        permissions: List[str],
        rate_limit: int = 100,
        expires_hours: Optional[int] = None,
    ) -> str:
        """Generate a new API key"""
        # Generate a secure random key
        key = secrets.token_urlsafe(32)

        expires = None
        if expires_hours:
            expires = (datetime.now() + timedelta(hours=expires_hours)).isoformat()

        self.api_keys[key] = {
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "expires": expires,
            "created": datetime.now().isoformat(),
        }

        self.key_usage[key] = []
        self.rate_limits[key] = rate_limit

        logger.info(f"Generated new API key: {name} with permissions: {permissions}")
        return key

    def validate_api_key(self, key: str) -> Tuple[bool, Optional[Dict]]:
        """Validate an API key"""
        if key not in self.api_keys:
            return False, None

        config = self.api_keys[key]

        # Check if key has expired
        if config["expires"]:
            expires = datetime.fromisoformat(config["expires"])
            if datetime.now() > expires:
                logger.warning(f"API key {key} has expired")
                return False, None

        return True, config

    def check_rate_limit(self, key: str) -> Tuple[bool, int, int]:
        """Check if API key is within rate limits"""
        if key not in self.api_keys:
            return False, 0, 0

        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Clean old usage records
        self.key_usage[key] = [
            usage_time for usage_time in self.key_usage[key] if usage_time > hour_ago
        ]

        current_usage = len(self.key_usage[key])
        rate_limit = self.rate_limits[key]

        return current_usage < rate_limit, current_usage, rate_limit

    def record_usage(self, key: str):
        """Record API key usage"""
        if key in self.key_usage:
            self.key_usage[key].append(datetime.now())

    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key"""
        if key in self.api_keys:
            del self.api_keys[key]
            del self.key_usage[key]
            if key in self.rate_limits:
                del self.rate_limits[key]
            logger.info(f"Revoked API key: {key}")
            return True
        return False

    def get_key_info(self, key: str) -> Optional[Dict]:
        """Get information about an API key (without exposing the key itself)"""
        if key not in self.api_keys:
            return None

        config = self.api_keys[key]
        return {
            "name": config["name"],
            "permissions": config["permissions"],
            "rate_limit": config["rate_limit"],
            "expires": config["expires"],
            "created": config["created"],
        }


# Global API key manager
api_key_manager = APIKeyManager()


def require_api_key(permissions: Optional[List[str]] = None):
    """Decorator to require API key authentication"""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get API key from header
            api_key = request.headers.get("X-API-Key")

            if not api_key:
                logger.warning(
                    f"API request without API key from {request.remote_addr}"
                )
                return (
                    jsonify(
                        {
                            "error": "API key required",
                            "message": "Please provide X-API-Key header",
                        }
                    ),
                    401,
                )

            # Validate API key
            is_valid, config = api_key_manager.validate_api_key(api_key)
            if not is_valid:
                logger.warning(f"Invalid API key from {request.remote_addr}")
                return (
                    jsonify(
                        {
                            "error": "Invalid API key",
                            "message": "The provided API key is invalid or expired",
                        }
                    ),
                    401,
                )

            # Check permissions
            if permissions:
                user_permissions = config["permissions"]
                if not any(perm in user_permissions for perm in permissions):
                    logger.warning(
                        f"Insufficient permissions for API key from {request.remote_addr}"
                    )
                    return (
                        jsonify(
                            {
                                "error": "Insufficient permissions",
                                "message": f"Required permissions: {permissions}",
                            }
                        ),
                        403,
                    )

            # Check rate limit
            within_limit, current_usage, rate_limit = api_key_manager.check_rate_limit(
                api_key
            )
            if not within_limit:
                logger.warning(
                    f"Rate limit exceeded for API key from {request.remote_addr}"
                )
                return (
                    jsonify(
                        {
                            "error": "Rate limit exceeded",
                            "message": f"Rate limit: {rate_limit} requests per hour",
                            "current_usage": current_usage,
                        }
                    ),
                    429,
                )

            # Record usage
            api_key_manager.record_usage(api_key)

            # Store user info in Flask g for use in the endpoint
            g.api_key = api_key
            g.user_permissions = config["permissions"]
            g.user_name = config["name"]

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_permission(permission: str):
    """Decorator to require specific permission"""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, "user_permissions"):
                return jsonify({"error": "Authentication required"}), 401

            if permission not in g.user_permissions:
                logger.warning(f"Permission denied for {g.user_name}: {permission}")
                return (
                    jsonify(
                        {
                            "error": "Permission denied",
                            "message": f"Required permission: {permission}",
                        }
                    ),
                    403,
                )

            return f(*args, **kwargs)

        return decorated_function

    return decorator


class SecurityMiddleware:
    """Security middleware for additional protection"""

    def __init__(self):
        self.blocked_ips = set()
        self.suspicious_ips = {}
        self.max_requests_per_minute = 60
        self.request_counts = {}

    def check_ip_security(self, ip: str) -> Tuple[bool, str]:
        """Check if IP is blocked or suspicious"""
        if ip in self.blocked_ips:
            return False, "IP is blocked"

        # Check for suspicious activity
        now = time.time()
        minute_ago = now - 60

        if ip not in self.request_counts:
            self.request_counts[ip] = []

        # Clean old requests
        self.request_counts[ip] = [
            req_time for req_time in self.request_counts[ip] if req_time > minute_ago
        ]

        # Check rate limit
        if len(self.request_counts[ip]) >= self.max_requests_per_minute:
            self.suspicious_ips[ip] = now
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            return False, "Too many requests"

        # Record request
        self.request_counts[ip].append(now)

        return True, "OK"

    def block_ip(self, ip: str, reason: str = "Manual block"):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        logger.warning(f"Blocked IP {ip}: {reason}")

    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        logger.info(f"Unblocked IP {ip}")

    def get_security_stats(self) -> Dict:
        """Get security statistics"""
        return {
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "total_ips_tracked": len(self.request_counts),
        }


# Global security middleware
security_middleware = SecurityMiddleware()


def security_check():
    """Security check middleware"""
    ip = request.remote_addr

    # Check IP security
    is_safe, message = security_middleware.check_ip_security(ip)
    if not is_safe:
        logger.warning(f"Security check failed for {ip}: {message}")
        return jsonify({"error": "Security check failed", "message": message}), 403

    return None


def generate_secure_token(data: Dict) -> str:
    """Generate a secure token for data"""
    # Create a timestamp
    timestamp = str(int(time.time()))

    # Create payload
    payload = {"data": data, "timestamp": timestamp}

    # Create signature
    secret = "your-secret-key"  # In production, use environment variable
    message = json.dumps(payload, sort_keys=True)
    signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    # Combine payload and signature
    token_data = {"payload": payload, "signature": signature}

    return json.dumps(token_data)


def verify_secure_token(token: str) -> Tuple[bool, Optional[Dict]]:
    """Verify a secure token"""
    try:
        token_data = json.loads(token)
        payload = token_data["payload"]
        signature = token_data["signature"]

        # Verify signature
        secret = "your-secret-key"  # In production, use environment variable
        message = json.dumps(payload, sort_keys=True)
        expected_signature = hmac.new(
            secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            return False, None

        # Check timestamp (token expires after 1 hour)
        timestamp = int(payload["timestamp"])
        if time.time() - timestamp > 3600:
            return False, None

        return True, payload["data"]

    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return False, None


# Convenience functions
def get_current_user() -> Optional[Dict]:
    """Get current authenticated user info"""
    if hasattr(g, "api_key"):
        return {
            "api_key": g.api_key,
            "name": g.user_name,
            "permissions": g.user_permissions,
        }
    return None


def has_permission(permission: str) -> bool:
    """Check if current user has specific permission"""
    if hasattr(g, "user_permissions"):
        return permission in g.user_permissions
    return False


def is_admin() -> bool:
    """Check if current user is admin"""
    return has_permission("admin")


def get_api_key_info(key: str) -> Optional[Dict]:
    """Get API key information"""
    return api_key_manager.get_key_info(key)


def create_api_key(name: str, permissions: List[str], rate_limit: int = 100) -> str:
    """Create a new API key"""
    return api_key_manager.generate_api_key(name, permissions, rate_limit)


def revoke_api_key(key: str) -> bool:
    """Revoke an API key"""
    return api_key_manager.revoke_api_key(key)
