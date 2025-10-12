#!/usr/bin/env python3
"""
Automated Data Population Script for AI Cybersecurity Tool
Simulates realistic network traffic and threats to populate the dashboard
"""

import requests
import json
import time
import random
import logging
from datetime import datetime
import threading
import signal
import sys
import os

# API Configuration
API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_populator.log'),
        logging.StreamHandler()
    ]
)

class AutoDataPopulator:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY
        })
        self.running = True
        self.stats = {
            'total_requests': 0,
            'threats_detected': 0,
            'benign_traffic': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
    def check_api_health(self):
        """Check if API is running and healthy"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"API health check failed: {e}")
            return False
    
    def send_traffic_data(self, traffic_data, traffic_type):
        """Send traffic data to API and return response"""
        try:
            response = self.session.post(
                f"{API_BASE_URL}/predict", 
                json=traffic_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.stats['total_requests'] += 1
                
                if data.get('threat_detected', False):
                    self.stats['threats_detected'] += 1
                else:
                    self.stats['benign_traffic'] += 1
                
                threat_score = data.get('threat_score', 0)
                threat_level = data.get('threat_level', 'Unknown')
                processing_time = data.get('processing_time_ms', 0)
                
                logging.info(f"{traffic_type}: Score={threat_score:.3f} | Level={threat_level} | Time={processing_time:.1f}ms")
                
                return data
            else:
                logging.error(f"{traffic_type} failed: HTTP {response.status_code}")
                self.stats['errors'] += 1
                return None
                
        except Exception as e:
            logging.error(f"{traffic_type} error: {e}")
            self.stats['errors'] += 1
            return None
    
    def generate_benign_traffic(self):
        """Generate realistic benign network traffic"""
        base_features = {
            "Destination Port": random.choice([80, 443, 22, 21, 25, 53]),
            "Flow Duration": random.randint(1000, 10000),
            "Total Fwd Packets": random.randint(10, 100),
            "Total Backward Packets": random.randint(8, 90),
            "Total Length of Fwd Packets": random.randint(1000, 10000),
            "Total Length of Bwd Packets": random.randint(800, 9000),
            "Fwd Packet Length Max": random.randint(1000, 1500),
            "Fwd Packet Length Min": random.randint(64, 128),
            "Fwd Packet Length Mean": random.randint(100, 200),
            "Fwd Packet Length Std": random.randint(50, 150),
            "Bwd Packet Length Max": random.randint(1000, 1500),
            "Bwd Packet Length Min": random.randint(64, 128),
            "Bwd Packet Length Mean": random.randint(100, 200),
            "Bwd Packet Length Std": random.randint(50, 150),
            "Flow Bytes/s": random.randint(500, 5000),
            "Flow Packets/s": random.randint(5, 50),
            "Flow IAT Mean": random.randint(50, 200),
            "Flow IAT Std": random.randint(10, 50),
            "Flow IAT Max": random.randint(100, 500),
            "Flow IAT Min": random.randint(10, 100),
            "Fwd IAT Total": random.randint(500, 5000),
            "Fwd IAT Mean": random.randint(50, 200),
            "Fwd IAT Std": random.randint(10, 50),
            "Fwd IAT Max": random.randint(100, 500),
            "Fwd IAT Min": random.randint(10, 100),
            "Bwd IAT Total": random.randint(400, 4500),
            "Bwd IAT Mean": random.randint(50, 200),
            "Bwd IAT Std": random.randint(10, 50),
            "Bwd IAT Max": random.randint(100, 500),
            "Bwd IAT Min": random.randint(10, 100),
            "Fwd PSH Flags": random.randint(0, 10),
            "Bwd PSH Flags": random.randint(0, 10),
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,
            "Bwd Header Length": 20,
            "Fwd Packets/s": random.randint(5, 25),
            "Bwd Packets/s": random.randint(4, 20),
            "Min Packet Length": random.randint(64, 128),
            "Max Packet Length": random.randint(1000, 1500),
            "Packet Length Mean": random.randint(100, 200),
            "Packet Length Std": random.randint(50, 150),
            "Packet Length Variance": random.randint(2500, 22500),
            "FIN Flag Count": random.randint(0, 2),
            "SYN Flag Count": 1,
            "RST Flag Count": random.randint(0, 1),
            "PSH Flag Count": random.randint(0, 20),
            "ACK Flag Count": random.randint(10, 50),
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": random.uniform(0.7, 1.0),
            "Average Packet Size": random.randint(100, 200),
            "Avg Fwd Segment Size": random.randint(100, 200),
            "Avg Bwd Segment Size": random.randint(100, 200),
            "Fwd Header Length.1": 20,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": random.randint(10, 100),
            "Subflow Fwd Bytes": random.randint(1000, 10000),
            "Subflow Bwd Packets": random.randint(8, 90),
            "Subflow Bwd Bytes": random.randint(800, 9000),
            "Init_Win_bytes_forward": 65535,
            "Init_Win_bytes_backward": 65535,
            "act_data_pkt_fwd": random.randint(0, 20),
            "min_seg_size_forward": 0,
            "Active Mean": random.randint(50, 200),
            "Active Std": random.randint(10, 50),
            "Active Max": random.randint(100, 500),
            "Active Min": random.randint(10, 100),
            "Idle Mean": random.randint(0, 100),
            "Idle Std": random.randint(0, 50),
            "Idle Max": random.randint(0, 200),
            "Idle Min": 0
        }
        
        return {
            "features": base_features,
            "source_ip": f"192.168.1.{random.randint(10, 254)}",
            "attack_type": "Benign"
        }
    
    def generate_ddos_attack(self):
        """Generate DDoS attack traffic"""
        attack_features = {
            "Destination Port": 80,
            "Flow Duration": random.randint(10, 100),  # Very short
            "Total Fwd Packets": random.randint(5000, 50000),  # Extremely high
            "Total Backward Packets": random.randint(10, 100),  # Very few responses
            "Total Length of Fwd Packets": random.randint(500000, 5000000),
            "Total Length of Bwd Packets": random.randint(1000, 10000),
            "Fwd Packet Length Max": 1500,
            "Fwd Packet Length Min": 64,
            "Fwd Packet Length Mean": random.randint(80, 120),
            "Fwd Packet Length Std": random.randint(50, 100),
            "Bwd Packet Length Max": random.randint(50, 150),
            "Bwd Packet Length Min": 0,
            "Bwd Packet Length Mean": random.randint(10, 50),
            "Bwd Packet Length Std": random.randint(5, 25),
            "Flow Bytes/s": random.randint(10000000, 100000000),  # Extremely high
            "Flow Packets/s": random.randint(100000, 1000000),  # Extremely high
            "Flow IAT Mean": random.uniform(0.1, 1.0),  # Very short intervals
            "Flow IAT Std": random.uniform(0.01, 0.1),
            "Flow IAT Max": random.uniform(0.5, 2.0),
            "Flow IAT Min": random.uniform(0.001, 0.1),
            "Fwd IAT Total": random.randint(10, 100),
            "Fwd IAT Mean": random.uniform(0.001, 0.01),
            "Fwd IAT Std": random.uniform(0.0001, 0.001),
            "Fwd IAT Max": random.uniform(0.005, 0.02),
            "Fwd IAT Min": random.uniform(0.0001, 0.002),
            "Bwd IAT Total": random.randint(50, 500),
            "Bwd IAT Mean": random.uniform(0.5, 2.0),
            "Bwd IAT Std": random.uniform(0.1, 0.5),
            "Bwd IAT Max": random.uniform(1.0, 5.0),
            "Bwd IAT Min": random.uniform(0.1, 1.0),
            "Fwd PSH Flags": 0,
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,
            "Bwd Header Length": 20,
            "Fwd Packets/s": random.randint(100000, 1000000),
            "Bwd Packets/s": random.randint(100, 1000),
            "Min Packet Length": 64,
            "Max Packet Length": 1500,
            "Packet Length Mean": random.randint(80, 120),
            "Packet Length Std": random.randint(50, 100),
            "Packet Length Variance": random.randint(2500, 10000),
            "FIN Flag Count": 0,
            "SYN Flag Count": random.randint(1000, 10000),
            "RST Flag Count": 0,
            "PSH Flag Count": 0,
            "ACK Flag Count": random.randint(10, 100),
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": random.uniform(0.001, 0.1),  # Very low response ratio
            "Average Packet Size": random.randint(80, 120),
            "Avg Fwd Segment Size": random.randint(80, 120),
            "Avg Bwd Segment Size": random.randint(10, 50),
            "Fwd Header Length.1": 20,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": random.randint(5000, 50000),
            "Subflow Fwd Bytes": random.randint(500000, 5000000),
            "Subflow Bwd Packets": random.randint(10, 100),
            "Subflow Bwd Bytes": random.randint(1000, 10000),
            "Init_Win_bytes_forward": 65535,
            "Init_Win_bytes_backward": 0,  # No response window
            "act_data_pkt_fwd": 0,
            "min_seg_size_forward": 0,
            "Active Mean": 0,
            "Active Std": 0,
            "Active Max": 0,
            "Active Min": 0,
            "Idle Mean": 0,
            "Idle Std": 0,
            "Idle Max": 0,
            "Idle Min": 0
        }
        
        return {
            "features": attack_features,
            "source_ip": f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "attack_type": "DDoS"
        }
    
    def generate_port_scan(self):
        """Generate port scan attack traffic"""
        scan_features = {
            "Destination Port": random.randint(1, 65535),
            "Flow Duration": random.randint(50, 200),
            "Total Fwd Packets": 1,
            "Total Backward Packets": 1,
            "Total Length of Fwd Packets": 60,
            "Total Length of Bwd Packets": 60,
            "Fwd Packet Length Max": 60,
            "Fwd Packet Length Min": 60,
            "Fwd Packet Length Mean": 60,
            "Fwd Packet Length Std": 0,
            "Bwd Packet Length Max": 60,
            "Bwd Packet Length Min": 60,
            "Bwd Packet Length Mean": 60,
            "Bwd Packet Length Std": 0,
            "Flow Bytes/s": random.randint(300, 600),
            "Flow Packets/s": random.randint(5, 10),
            "Flow IAT Mean": random.randint(50, 200),
            "Flow IAT Std": 0,
            "Flow IAT Max": random.randint(50, 200),
            "Flow IAT Min": random.randint(50, 200),
            "Fwd IAT Total": random.randint(50, 200),
            "Fwd IAT Mean": random.randint(50, 200),
            "Fwd IAT Std": 0,
            "Fwd IAT Max": random.randint(50, 200),
            "Fwd IAT Min": random.randint(50, 200),
            "Bwd IAT Total": random.randint(50, 200),
            "Bwd IAT Mean": random.randint(50, 200),
            "Bwd IAT Std": 0,
            "Bwd IAT Max": random.randint(50, 200),
            "Bwd IAT Min": random.randint(50, 200),
            "Fwd PSH Flags": 0,
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,
            "Bwd Header Length": 20,
            "Fwd Packets/s": random.randint(5, 10),
            "Bwd Packets/s": random.randint(5, 10),
            "Min Packet Length": 60,
            "Max Packet Length": 60,
            "Packet Length Mean": 60,
            "Packet Length Std": 0,
            "Packet Length Variance": 0,
            "FIN Flag Count": 0,
            "SYN Flag Count": 1,
            "RST Flag Count": 0,
            "PSH Flag Count": 0,
            "ACK Flag Count": 0,
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": 1.0,
            "Average Packet Size": 60,
            "Avg Fwd Segment Size": 60,
            "Avg Bwd Segment Size": 60,
            "Fwd Header Length.1": 20,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": 1,
            "Subflow Fwd Bytes": 60,
            "Subflow Bwd Packets": 1,
            "Subflow Bwd Bytes": 60,
            "Init_Win_bytes_forward": 65535,
            "Init_Win_bytes_backward": 65535,
            "act_data_pkt_fwd": 0,
            "min_seg_size_forward": 0,
            "Active Mean": 0,
            "Active Std": 0,
            "Active Max": 0,
            "Active Min": 0,
            "Idle Mean": 0,
            "Idle Std": 0,
            "Idle Max": 0,
            "Idle Min": 0
        }
        
        return {
            "features": scan_features,
            "source_ip": f"172.16.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "attack_type": "Port Scan"
        }
    
    def generate_brute_force(self):
        """Generate brute force attack traffic"""
        brute_features = {
            "Destination Port": random.choice([22, 23, 21, 25, 110, 143]),
            "Flow Duration": random.randint(2000, 10000),
            "Total Fwd Packets": random.randint(50, 200),
            "Total Backward Packets": random.randint(50, 200),
            "Total Length of Fwd Packets": random.randint(5000, 20000),
            "Total Length of Bwd Packets": random.randint(2500, 10000),
            "Fwd Packet Length Max": random.randint(100, 300),
            "Fwd Packet Length Min": random.randint(50, 100),
            "Fwd Packet Length Mean": random.randint(100, 150),
            "Fwd Packet Length Std": random.randint(20, 50),
            "Bwd Packet Length Max": random.randint(50, 150),
            "Bwd Packet Length Min": random.randint(20, 50),
            "Bwd Packet Length Mean": random.randint(50, 100),
            "Bwd Packet Length Std": random.randint(10, 30),
            "Flow Bytes/s": random.randint(1000, 3000),
            "Flow Packets/s": random.randint(10, 30),
            "Flow IAT Mean": random.randint(30, 100),
            "Flow IAT Std": random.randint(5, 20),
            "Flow IAT Max": random.randint(100, 300),
            "Flow IAT Min": random.randint(20, 50),
            "Fwd IAT Total": random.randint(2000, 10000),
            "Fwd IAT Mean": random.randint(30, 100),
            "Fwd IAT Std": random.randint(5, 20),
            "Fwd IAT Max": random.randint(100, 300),
            "Fwd IAT Min": random.randint(20, 50),
            "Bwd IAT Total": random.randint(2000, 10000),
            "Bwd IAT Mean": random.randint(30, 100),
            "Bwd IAT Std": random.randint(5, 20),
            "Bwd IAT Max": random.randint(100, 300),
            "Bwd IAT Min": random.randint(20, 50),
            "Fwd PSH Flags": random.randint(50, 200),
            "Bwd PSH Flags": random.randint(50, 200),
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,
            "Bwd Header Length": 20,
            "Fwd Packets/s": random.randint(10, 30),
            "Bwd Packets/s": random.randint(10, 30),
            "Min Packet Length": random.randint(20, 50),
            "Max Packet Length": random.randint(100, 300),
            "Packet Length Mean": random.randint(80, 150),
            "Packet Length Std": random.randint(20, 50),
            "Packet Length Variance": random.randint(400, 2500),
            "FIN Flag Count": 0,
            "SYN Flag Count": 1,
            "RST Flag Count": 0,
            "PSH Flag Count": random.randint(100, 400),
            "ACK Flag Count": random.randint(100, 400),
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": random.uniform(0.3, 0.7),
            "Average Packet Size": random.randint(80, 150),
            "Avg Fwd Segment Size": random.randint(100, 150),
            "Avg Bwd Segment Size": random.randint(50, 100),
            "Fwd Header Length.1": 20,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": random.randint(50, 200),
            "Subflow Fwd Bytes": random.randint(5000, 20000),
            "Subflow Bwd Packets": random.randint(50, 200),
            "Subflow Bwd Bytes": random.randint(2500, 10000),
            "Init_Win_bytes_forward": 65535,
            "Init_Win_bytes_backward": 65535,
            "act_data_pkt_fwd": random.randint(50, 200),
            "min_seg_size_forward": 0,
            "Active Mean": random.randint(30, 100),
            "Active Std": random.randint(5, 20),
            "Active Max": random.randint(100, 300),
            "Active Min": random.randint(20, 50),
            "Idle Mean": 0,
            "Idle Std": 0,
            "Idle Max": 0,
            "Idle Min": 0
        }
        
        return {
            "features": brute_features,
            "source_ip": f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "attack_type": "Brute Force"
        }
    
    def print_stats(self):
        """Print current statistics"""
        runtime = datetime.now() - self.stats['start_time']
        runtime_minutes = runtime.total_seconds() / 60
        
        print(f"\n📊 AUTO POPULATOR STATISTICS")
        print(f"=" * 40)
        print(f"⏱️  Runtime: {runtime_minutes:.1f} minutes")
        print(f"📡 Total Requests: {self.stats['total_requests']}")
        print(f"🚨 Threats Detected: {self.stats['threats_detected']}")
        print(f"✅ Benign Traffic: {self.stats['benign_traffic']}")
        print(f"❌ Errors: {self.stats['errors']}")
        
        if self.stats['total_requests'] > 0:
            threat_rate = (self.stats['threats_detected'] / self.stats['total_requests']) * 100
            error_rate = (self.stats['errors'] / self.stats['total_requests']) * 100
            print(f"🎯 Threat Detection Rate: {threat_rate:.1f}%")
            print(f"⚠️  Error Rate: {error_rate:.1f}%")
        
        if runtime_minutes > 0:
            req_per_min = self.stats['total_requests'] / runtime_minutes
            print(f"📈 Requests/Minute: {req_per_min:.1f}")
    
    def run_continuous_population(self, interval_seconds=5, threat_ratio=0.3):
        """
        Run continuous data population
        
        Args:
            interval_seconds: Time between requests
            threat_ratio: Ratio of threat traffic (0.0 = all benign, 1.0 = all threats)
        """
        logging.info(f"Starting continuous population with {interval_seconds}s interval, {threat_ratio*100:.1f}% threats")
        
        while self.running:
            try:
                # Check API health first
                if not self.check_api_health():
                    logging.error("API is not responding, waiting 30 seconds...")
                    time.sleep(30)
                    continue
                
                # Determine traffic type based on threat ratio
                if random.random() < threat_ratio:
                    # Generate threat traffic
                    attack_type = random.choice(['ddos', 'port_scan', 'brute_force'])
                    
                    if attack_type == 'ddos':
                        traffic_data = self.generate_ddos_attack()
                    elif attack_type == 'port_scan':
                        traffic_data = self.generate_port_scan()
                    else:  # brute_force
                        traffic_data = self.generate_brute_force()
                        
                    traffic_type = f"Threat ({attack_type.upper()})"
                else:
                    # Generate benign traffic
                    traffic_data = self.generate_benign_traffic()
                    traffic_type = "Benign Traffic"
                
                # Send traffic data
                result = self.send_traffic_data(traffic_data, traffic_type)
                
                # Print stats every 10 requests
                if self.stats['total_requests'] % 10 == 0:
                    self.print_stats()
                
                # Wait before next request
                time.sleep(interval_seconds)
                
            except Exception as e:
                logging.error(f"Error in continuous population: {e}")
                time.sleep(5)  # Wait before retrying
        
        logging.info("Continuous population stopped")
        self.print_stats()

def main():
    """Main function"""
    print("🚀 AI CYBERSECURITY TOOL - AUTO DATA POPULATOR")
    print("=" * 50)
    print("📊 This will continuously populate the dashboard with realistic data")
    print("🔄 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Check if API is running
    populator = AutoDataPopulator()
    if not populator.check_api_health():
        print("❌ API is not running. Please start the API server first:")
        print("   python api/app.py")
        return
    
    print("✅ API is healthy, starting population...")
    
    # Get user preferences
    try:
        interval = input("Enter interval between requests in seconds (default: 5): ").strip()
        interval = int(interval) if interval else 5
        
        threat_ratio = input("Enter threat ratio (0.0-1.0, default: 0.3): ").strip()
        threat_ratio = float(threat_ratio) if threat_ratio else 0.3
        
        print(f"\n🎯 Starting with {interval}s interval, {threat_ratio*100:.1f}% threats")
        print("🔄 Population started! Check your dashboard at http://localhost:3000")
        
        # Start continuous population
        populator.run_continuous_population(interval, threat_ratio)
        
    except KeyboardInterrupt:
        print("\n⏹️ Population stopped by user")
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Auto population completed!")
    print("📊 Check your dashboard for final statistics!")

if __name__ == "__main__":
    main()
