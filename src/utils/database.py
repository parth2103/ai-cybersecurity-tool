#!/usr/bin/env python3
"""
Database integration for AI Cybersecurity Tool
Provides persistent storage for threats, alerts, and system data
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading

from .logger import get_logger
from .error_handler import handle_data_error

logger = get_logger("database")

class ThreatDatabase:
    """SQLite database for threat detection data"""
    
    def __init__(self, db_path: str = "data/threats.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Threats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    threat_score REAL NOT NULL,
                    source_ip TEXT,
                    attack_type TEXT,
                    model_predictions TEXT,
                    processing_time_ms REAL,
                    user_agent TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    threat_level TEXT NOT NULL,
                    threat_score REAL NOT NULL,
                    source_ip TEXT,
                    attack_type TEXT,
                    message TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage REAL,
                    active_connections INTEGER,
                    queue_size INTEGER,
                    total_predictions INTEGER,
                    threats_detected INTEGER,
                    detection_rate REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # API usage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    api_key TEXT,
                    endpoint TEXT,
                    method TEXT,
                    response_time_ms REAL,
                    status_code INTEGER,
                    client_ip TEXT,
                    user_agent TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    inference_time_ms REAL,
                    memory_usage_mb REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_threats_timestamp ON threats(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_threats_threat_level ON threats(threat_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_threats_source_ip ON threats(source_ip)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_threat_level ON alerts(threat_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database initialized at {self.db_path}")
    
    def store_threat(self, threat_data: Dict[str, Any]) -> int:
        """Store threat detection data"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO threats (
                        timestamp, threat_level, threat_score, source_ip,
                        attack_type, model_predictions, processing_time_ms, user_agent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    threat_data.get('timestamp', datetime.now().isoformat()),
                    threat_data.get('threat_level', 'Unknown'),
                    threat_data.get('threat_score', 0.0),
                    threat_data.get('source_ip', 'Unknown'),
                    threat_data.get('attack_type', 'Unknown'),
                    json.dumps(threat_data.get('model_predictions', {})),
                    threat_data.get('processing_time', 0.0),
                    threat_data.get('user_agent', 'Unknown')
                ))
                
                threat_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                logger.debug(f"Stored threat with ID: {threat_id}")
                return threat_id
                
        except Exception as e:
            handle_data_error(e, "threat storage")
            raise
    
    def store_alert(self, alert_data: Dict[str, Any]) -> int:
        """Store alert data"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO alerts (
                        timestamp, threat_level, threat_score, source_ip,
                        attack_type, message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert_data.get('timestamp', datetime.now().isoformat()),
                    alert_data.get('threat_level', 'Unknown'),
                    alert_data.get('threat_score', 0.0),
                    alert_data.get('source_ip', 'Unknown'),
                    alert_data.get('attack_type', 'Unknown'),
                    alert_data.get('message', '')
                ))
                
                alert_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                logger.debug(f"Stored alert with ID: {alert_id}")
                return alert_id
                
        except Exception as e:
            handle_data_error(e, "alert storage")
            raise
    
    def store_system_metrics(self, metrics_data: Dict[str, Any]) -> int:
        """Store system metrics"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, disk_usage,
                        active_connections, queue_size, total_predictions,
                        threats_detected, detection_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics_data.get('timestamp', datetime.now().isoformat()),
                    metrics_data.get('cpu_percent', 0.0),
                    metrics_data.get('memory_percent', 0.0),
                    metrics_data.get('disk_usage', 0.0),
                    metrics_data.get('active_connections', 0),
                    metrics_data.get('queue_size', 0),
                    metrics_data.get('total_predictions', 0),
                    metrics_data.get('threats_detected', 0),
                    metrics_data.get('detection_rate', 0.0)
                ))
                
                metrics_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                logger.debug(f"Stored system metrics with ID: {metrics_id}")
                return metrics_id
                
        except Exception as e:
            handle_data_error(e, "system metrics storage")
            raise
    
    def store_api_usage(self, usage_data: Dict[str, Any]) -> int:
        """Store API usage data"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO api_usage (
                        timestamp, api_key, endpoint, method, response_time_ms,
                        status_code, client_ip, user_agent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    usage_data.get('timestamp', datetime.now().isoformat()),
                    usage_data.get('api_key', 'Unknown'),
                    usage_data.get('endpoint', 'Unknown'),
                    usage_data.get('method', 'Unknown'),
                    usage_data.get('response_time', 0.0),
                    usage_data.get('status_code', 0),
                    usage_data.get('client_ip', 'Unknown'),
                    usage_data.get('user_agent', 'Unknown')
                ))
                
                usage_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                logger.debug(f"Stored API usage with ID: {usage_id}")
                return usage_id
                
        except Exception as e:
            handle_data_error(e, "API usage storage")
            raise
    
    def store_model_performance(self, performance_data: Dict[str, Any]) -> int:
        """Store model performance data"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO model_performance (
                        timestamp, model_name, accuracy, precision, recall,
                        f1_score, inference_time_ms, memory_usage_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance_data.get('timestamp', datetime.now().isoformat()),
                    performance_data.get('model_name', 'Unknown'),
                    performance_data.get('accuracy', 0.0),
                    performance_data.get('precision', 0.0),
                    performance_data.get('recall', 0.0),
                    performance_data.get('f1_score', 0.0),
                    performance_data.get('inference_time', 0.0),
                    performance_data.get('memory_usage', 0.0)
                ))
                
                perf_id = cursor.lastrowid
                conn.commit()
                conn.close()
                
                logger.debug(f"Stored model performance with ID: {perf_id}")
                return perf_id
                
        except Exception as e:
            handle_data_error(e, "model performance storage")
            raise
    
    def get_threats(self, limit: int = 100, threat_level: Optional[str] = None,
                   start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Get threats with optional filtering"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = 'SELECT * FROM threats WHERE 1=1'
                params = []
                
                if threat_level:
                    query += ' AND threat_level = ?'
                    params.append(threat_level)
                
                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                threats = []
                for row in rows:
                    threat = dict(zip(columns, row))
                    # Parse JSON fields
                    if threat['model_predictions']:
                        threat['model_predictions'] = json.loads(threat['model_predictions'])
                    threats.append(threat)
                
                conn.close()
                return threats
                
        except Exception as e:
            handle_data_error(e, "threat retrieval")
            raise
    
    def get_alerts(self, limit: int = 100, acknowledged: Optional[bool] = None) -> List[Dict]:
        """Get alerts with optional filtering"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = 'SELECT * FROM alerts WHERE 1=1'
                params = []
                
                if acknowledged is not None:
                    query += ' AND acknowledged = ?'
                    params.append(acknowledged)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                alerts = [dict(zip(columns, row)) for row in rows]
                conn.close()
                return alerts
                
        except Exception as e:
            handle_data_error(e, "alert retrieval")
            raise
    
    def get_system_metrics(self, limit: int = 100, hours: Optional[int] = None) -> List[Dict]:
        """Get system metrics with optional time filtering"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = 'SELECT * FROM system_metrics WHERE 1=1'
                params = []
                
                if hours:
                    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                    query += ' AND timestamp >= ?'
                    params.append(cutoff)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                metrics = [dict(zip(columns, row)) for row in rows]
                conn.close()
                return metrics
                
        except Exception as e:
            handle_data_error(e, "system metrics retrieval")
            raise
    
    def get_threat_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get threat statistics for the last N days"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                
                # Total threats
                cursor.execute('SELECT COUNT(*) FROM threats WHERE timestamp >= ?', (cutoff,))
                total_threats = cursor.fetchone()[0]
                
                # Threats by level
                cursor.execute('''
                    SELECT threat_level, COUNT(*) 
                    FROM threats 
                    WHERE timestamp >= ? 
                    GROUP BY threat_level
                ''', (cutoff,))
                threats_by_level = dict(cursor.fetchall())
                
                # Average threat score
                cursor.execute('''
                    SELECT AVG(threat_score) 
                    FROM threats 
                    WHERE timestamp >= ?
                ''', (cutoff,))
                avg_threat_score = cursor.fetchone()[0] or 0.0
                
                # Top source IPs
                cursor.execute('''
                    SELECT source_ip, COUNT(*) as count
                    FROM threats 
                    WHERE timestamp >= ? AND source_ip != 'Unknown'
                    GROUP BY source_ip 
                    ORDER BY count DESC 
                    LIMIT 10
                ''', (cutoff,))
                top_source_ips = [{'ip': row[0], 'count': row[1]} for row in cursor.fetchall()]
                
                conn.close()
                
                return {
                    'total_threats': total_threats,
                    'threats_by_level': threats_by_level,
                    'average_threat_score': avg_threat_score,
                    'top_source_ips': top_source_ips,
                    'period_days': days
                }
                
        except Exception as e:
            handle_data_error(e, "threat statistics")
            raise
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE alerts 
                    SET acknowledged = TRUE, acknowledged_by = ?, acknowledged_at = ?
                    WHERE id = ?
                ''', (acknowledged_by, datetime.now().isoformat(), alert_id))
                
                success = cursor.rowcount > 0
                conn.commit()
                conn.close()
                
                if success:
                    logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                
                return success
                
        except Exception as e:
            handle_data_error(e, "alert acknowledgment")
            raise
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data to prevent database bloat"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                
                # Clean up old threats (keep only high/critical)
                cursor.execute('''
                    DELETE FROM threats 
                    WHERE timestamp < ? AND threat_level NOT IN ('High', 'Critical')
                ''', (cutoff,))
                threats_deleted = cursor.rowcount
                
                # Clean up old system metrics (keep daily averages)
                cursor.execute('''
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                ''', (cutoff,))
                metrics_deleted = cursor.rowcount
                
                # Clean up old API usage
                cursor.execute('''
                    DELETE FROM api_usage 
                    WHERE timestamp < ?
                ''', (cutoff,))
                usage_deleted = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                logger.info(f"Cleaned up old data: {threats_deleted} threats, {metrics_deleted} metrics, {usage_deleted} API usage records")
                
        except Exception as e:
            handle_data_error(e, "data cleanup")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                stats = {}
                
                # Table sizes
                tables = ['threats', 'alerts', 'system_metrics', 'api_usage', 'model_performance']
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                stats['database_size_bytes'] = db_size
                stats['database_size_mb'] = db_size / (1024 * 1024)
                
                conn.close()
                return stats
                
        except Exception as e:
            handle_data_error(e, "database statistics")
            raise

# Global database instance
threat_db = ThreatDatabase()

# Convenience functions
def store_threat(threat_data: Dict[str, Any]) -> int:
    """Store threat detection data"""
    return threat_db.store_threat(threat_data)

def store_alert(alert_data: Dict[str, Any]) -> int:
    """Store alert data"""
    return threat_db.store_alert(alert_data)

def store_system_metrics(metrics_data: Dict[str, Any]) -> int:
    """Store system metrics"""
    return threat_db.store_system_metrics(metrics_data)

def store_api_usage(usage_data: Dict[str, Any]) -> int:
    """Store API usage data"""
    return threat_db.store_api_usage(usage_data)

def get_threats(limit: int = 100, **filters) -> List[Dict]:
    """Get threats with filtering"""
    return threat_db.get_threats(limit, **filters)

def get_alerts(limit: int = 100, acknowledged: Optional[bool] = None) -> List[Dict]:
    """Get alerts with filtering"""
    return threat_db.get_alerts(limit, acknowledged)

def get_threat_statistics(days: int = 7) -> Dict[str, Any]:
    """Get threat statistics"""
    return threat_db.get_threat_statistics(days)
