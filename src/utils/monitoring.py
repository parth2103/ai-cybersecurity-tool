#!/usr/bin/env python3
"""
Monitoring and alerting system for AI Cybersecurity Tool
Provides system health monitoring, metrics collection, and alerting
"""

import time
import psutil
import threading
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from dataclasses import dataclass
from enum import Enum
import requests

from .logger import get_logger
from .error_handler import handle_data_error

logger = get_logger("monitoring")


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict] = None


class MetricsCollector:
    """Collects system and application metrics"""

    def __init__(self):
        self.metrics = {}
        self.collection_interval = 30  # seconds
        self.is_running = False
        self.collection_thread = None

    def start_collection(self):
        """Start metrics collection in background thread"""
        if self.is_running:
            return

        self.is_running = True
        self.collection_thread = threading.Thread(
            target=self._collect_metrics_loop, daemon=True
        )
        self.collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop metrics collection"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")

    def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.is_running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_free = disk.free / (1024**3)  # GB

            # Network metrics
            network = psutil.net_io_counters()

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            process_cpu = process.cpu_percent()

            self.metrics["system"] = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {"percent": cpu_percent, "count": cpu_count},
                "memory": {"percent": memory_percent, "available_gb": memory_available},
                "disk": {"percent": disk_percent, "free_gb": disk_free},
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
                "process": {"memory_mb": process_memory, "cpu_percent": process_cpu},
            }

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # This would be populated by the application
            # For now, we'll create a placeholder structure
            self.metrics["application"] = {
                "timestamp": datetime.now().isoformat(),
                "api_requests": 0,
                "threats_detected": 0,
                "alerts_generated": 0,
                "models_loaded": 0,
                "database_connections": 0,
            }

        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()

    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict]:
        """Get metric history (placeholder - would integrate with time series DB)"""
        # In a real implementation, this would query a time series database
        return []


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self):
        self.alerts = {}
        self.alert_handlers = []
        self.email_config = None
        self.webhook_config = None
        self.alert_rules = self._setup_default_rules()

    def _setup_default_rules(self) -> Dict[str, Dict]:
        """Setup default alerting rules"""
        return {
            "high_cpu": {
                "metric": "system.cpu.percent",
                "threshold": 80,
                "severity": AlertSeverity.HIGH,
                "message": "High CPU usage detected",
            },
            "high_memory": {
                "metric": "system.memory.percent",
                "threshold": 85,
                "severity": AlertSeverity.HIGH,
                "message": "High memory usage detected",
            },
            "low_disk_space": {
                "metric": "system.disk.percent",
                "threshold": 90,
                "severity": AlertSeverity.CRITICAL,
                "message": "Low disk space detected",
            },
            "api_errors": {
                "metric": "application.api_errors",
                "threshold": 10,
                "severity": AlertSeverity.MEDIUM,
                "message": "High number of API errors",
            },
            "threat_detection_rate": {
                "metric": "application.threat_detection_rate",
                "threshold": 0.1,  # 10%
                "severity": AlertSeverity.HIGH,
                "message": "High threat detection rate",
            },
        }

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
    ):
        """Configure email notifications"""
        self.email_config = {
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_email": from_email,
            "to_emails": to_emails,
        }
        logger.info("Email notifications configured")

    def configure_webhook(self, webhook_url: str, headers: Optional[Dict] = None):
        """Configure webhook notifications"""
        self.webhook_config = {"url": webhook_url, "headers": headers or {}}
        logger.info("Webhook notifications configured")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)

    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict] = None,
    ) -> Alert:
        """Create a new alert"""
        alert_id = f"{source}_{int(time.time())}"
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata,
        )

        self.alerts[alert_id] = alert
        self._process_alert(alert)

        logger.warning(f"Alert created: {title} ({severity.value})")
        return alert

    def _process_alert(self, alert: Alert):
        """Process alert through all handlers"""
        # Send email notification
        if self.email_config:
            self._send_email_alert(alert)

        # Send webhook notification
        if self.webhook_config:
            self._send_webhook_alert(alert)

        # Call custom handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def _send_email_alert(self, alert: Alert):
        """Send email notification"""
        try:
            msg = MimeMultipart()
            msg["From"] = self.email_config["from_email"]
            msg["To"] = ", ".join(self.email_config["to_emails"])
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            body = f"""
Alert Details:
- Title: {alert.title}
- Message: {alert.message}
- Severity: {alert.severity.value.upper()}
- Source: {alert.source}
- Timestamp: {alert.timestamp.isoformat()}
- Alert ID: {alert.id}

{json.dumps(alert.metadata, indent=2) if alert.metadata else ''}
"""

            msg.attach(MimeText(body, "plain"))

            server = smtplib.SMTP(
                self.email_config["smtp_host"], self.email_config["smtp_port"]
            )
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent: {alert.id}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_webhook_alert(self, alert: Alert):
        """Send webhook notification"""
        try:
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            }

            response = requests.post(
                self.webhook_config["url"],
                json=payload,
                headers=self.webhook_config["headers"],
                timeout=10,
            )

            if response.status_code == 200:
                logger.info(f"Webhook alert sent: {alert.id}")
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        return [alert for alert in self.alerts.values() if alert.severity == severity]


class HealthChecker:
    """Performs health checks on system components"""

    def __init__(self):
        self.checks = {}
        self.check_interval = 60  # seconds
        self.is_running = False
        self.check_thread = None
        self.alert_manager = None

    def set_alert_manager(self, alert_manager: AlertManager):
        """Set alert manager for health check notifications"""
        self.alert_manager = alert_manager

    def add_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        error_message: str,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
    ):
        """Add a health check"""
        self.checks[name] = {
            "function": check_func,
            "error_message": error_message,
            "severity": severity,
            "last_check": None,
            "last_result": None,
        }

    def start_checks(self):
        """Start health checks in background thread"""
        if self.is_running:
            return

        self.is_running = True
        self.check_thread = threading.Thread(target=self._run_checks_loop, daemon=True)
        self.check_thread.start()
        logger.info("Health checks started")

    def stop_checks(self):
        """Stop health checks"""
        self.is_running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        logger.info("Health checks stopped")

    def _run_checks_loop(self):
        """Main health check loop"""
        while self.is_running:
            try:
                for name, check in self.checks.items():
                    self._run_check(name, check)
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(self.check_interval)

    def _run_check(self, name: str, check: Dict):
        """Run a single health check"""
        try:
            result = check["function"]()
            check["last_check"] = datetime.now()
            check["last_result"] = result

            if not result and self.alert_manager:
                # Create alert for failed check
                self.alert_manager.create_alert(
                    title=f"Health Check Failed: {name}",
                    message=check["error_message"],
                    severity=check["severity"],
                    source="health_checker",
                    metadata={"check_name": name},
                )

        except Exception as e:
            logger.error(f"Health check {name} failed with exception: {e}")
            check["last_check"] = datetime.now()
            check["last_result"] = False

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        status = {
            "overall_healthy": True,
            "checks": {},
            "timestamp": datetime.now().isoformat(),
        }

        for name, check in self.checks.items():
            check_status = {
                "healthy": (
                    check["last_result"] if check["last_result"] is not None else None
                ),
                "last_check": (
                    check["last_check"].isoformat() if check["last_check"] else None
                ),
                "error_message": check["error_message"],
                "severity": check["severity"].value,
            }
            status["checks"][name] = check_status

            if check["last_result"] is False:
                status["overall_healthy"] = False

        return status


# Global instances
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
health_checker = HealthChecker()


# Convenience functions
def start_monitoring():
    """Start all monitoring components"""
    metrics_collector.start_collection()
    health_checker.start_checks()
    logger.info("Monitoring started")


def stop_monitoring():
    """Stop all monitoring components"""
    metrics_collector.stop_collection()
    health_checker.stop_checks()
    logger.info("Monitoring stopped")


def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status"""
    return {
        "metrics": metrics_collector.get_metrics(),
        "health": health_checker.get_health_status(),
        "alerts": {
            "active": len(alert_manager.get_active_alerts()),
            "critical": len(
                alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
            ),
            "high": len(alert_manager.get_alerts_by_severity(AlertSeverity.HIGH)),
        },
    }


def create_alert(
    title: str,
    message: str,
    severity: AlertSeverity,
    source: str,
    metadata: Optional[Dict] = None,
) -> Alert:
    """Create a new alert"""
    return alert_manager.create_alert(title, message, severity, source, metadata)


def configure_email_alerts(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    from_email: str,
    to_emails: List[str],
):
    """Configure email alerts"""
    alert_manager.configure_email(
        smtp_host, smtp_port, username, password, from_email, to_emails
    )


def configure_webhook_alerts(webhook_url: str, headers: Optional[Dict] = None):
    """Configure webhook alerts"""
    alert_manager.configure_webhook(webhook_url, headers)
