#!/usr/bin/env python3
"""
Live Threat Demo - Continuous threat simulation for dashboard testing
"""

import requests
import json
import time
import random
from datetime import datetime

API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"

def send_threat(attack_type, threat_data):
    """Send threat data and return response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            json=threat_data
        )
        
        if response.status_code == 200:
            data = response.json()
            threat_score = data.get('threat_score', 0)
            threat_level = data.get('threat_level', 'Unknown')
            
            print(f"🚨 {attack_type}: Score={threat_score:.3f} ({threat_score*100:.1f}%) | Level={threat_level}")
            return data
        else:
            print(f"❌ {attack_type} failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ {attack_type} error: {e}")
        return None

def create_ddos_attack():
    """Create DDoS attack data"""
    return {
        "features": {
            "Destination Port": 80,
            "Flow Duration": 50,
            "Total Fwd Packets": 10000,
            "Total Backward Packets": 100,
            "Total Length of Fwd Packets": 1000000,
            "Total Length of Bwd Packets": 10000,
            "Fwd Packet Length Max": 1500,
            "Fwd Packet Length Min": 64,
            "Fwd Packet Length Mean": 100,
            "Fwd Packet Length Std": 200,
            "Bwd Packet Length Max": 100,
            "Bwd Packet Length Min": 0,
            "Bwd Packet Length Mean": 10,
            "Bwd Packet Length Std": 20,
            "Flow Bytes/s": 20000000,
            "Flow Packets/s": 200000,
            "Flow IAT Mean": 0.5,
            "Flow IAT Std": 0.1,
            "Flow IAT Max": 1,
            "Flow IAT Min": 0.1,
            "Fwd IAT Total": 50,
            "Fwd IAT Mean": 0.005,
            "Fwd IAT Std": 0.001,
            "Fwd IAT Max": 0.01,
            "Fwd IAT Min": 0.001,
            "Bwd IAT Total": 100,
            "Bwd IAT Mean": 1,
            "Bwd IAT Std": 0.5,
            "Bwd IAT Max": 2,
            "Bwd IAT Min": 0.5,
            "Fwd PSH Flags": 0,
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,
            "Bwd Header Length": 20,
            "Fwd Packets/s": 200000,
            "Bwd Packets/s": 2000,
            "Min Packet Length": 64,
            "Max Packet Length": 1500,
            "Packet Length Mean": 100,
            "Packet Length Std": 200,
            "Packet Length Variance": 40000,
            "FIN Flag Count": 0,
            "SYN Flag Count": 10000,
            "RST Flag Count": 0,
            "PSH Flag Count": 0,
            "ACK Flag Count": 100,
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": 0.01,
            "Average Packet Size": 100,
            "Avg Fwd Segment Size": 100,
            "Avg Bwd Segment Size": 10,
            "Fwd Header Length.1": 20,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": 10000,
            "Subflow Fwd Bytes": 1000000,
            "Subflow Bwd Packets": 100,
            "Subflow Bwd Bytes": 10000,
            "Init_Win_bytes_forward": 65535,
            "Init_Win_bytes_backward": 0,
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
        },
        "source_ip": f"192.168.1.{random.randint(100, 200)}",
        "attack_type": "DDoS"
    }

def create_benign_traffic():
    """Create benign traffic data"""
    return {
        "features": {
            "Destination Port": 80,
            "Flow Duration": 2000,
            "Total Fwd Packets": 20,
            "Total Backward Packets": 18,
            "Total Length of Fwd Packets": 2000,
            "Total Length of Bwd Packets": 1800,
            "Fwd Packet Length Max": 1500,
            "Fwd Packet Length Min": 64,
            "Fwd Packet Length Mean": 100,
            "Fwd Packet Length Std": 200,
            "Bwd Packet Length Max": 1500,
            "Bwd Packet Length Min": 64,
            "Bwd Packet Length Mean": 100,
            "Bwd Packet Length Std": 200,
            "Flow Bytes/s": 1000,
            "Flow Packets/s": 10,
            "Flow IAT Mean": 100,
            "Flow IAT Std": 20,
            "Flow IAT Max": 200,
            "Flow IAT Min": 50,
            "Fwd IAT Total": 2000,
            "Fwd IAT Mean": 100,
            "Fwd IAT Std": 20,
            "Fwd IAT Max": 200,
            "Fwd IAT Min": 50,
            "Bwd IAT Total": 1800,
            "Bwd IAT Mean": 100,
            "Bwd IAT Std": 20,
            "Bwd IAT Max": 200,
            "Bwd IAT Min": 50,
            "Fwd PSH Flags": 5,
            "Bwd PSH Flags": 5,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,
            "Bwd Header Length": 20,
            "Fwd Packets/s": 10,
            "Bwd Packets/s": 9,
            "Min Packet Length": 64,
            "Max Packet Length": 1500,
            "Packet Length Mean": 100,
            "Packet Length Std": 200,
            "Packet Length Variance": 40000,
            "FIN Flag Count": 1,
            "SYN Flag Count": 1,
            "RST Flag Count": 0,
            "PSH Flag Count": 10,
            "ACK Flag Count": 20,
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": 0.9,
            "Average Packet Size": 100,
            "Avg Fwd Segment Size": 100,
            "Avg Bwd Segment Size": 100,
            "Fwd Header Length.1": 20,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": 20,
            "Subflow Fwd Bytes": 2000,
            "Subflow Bwd Packets": 18,
            "Subflow Bwd Bytes": 1800,
            "Init_Win_bytes_forward": 65535,
            "Init_Win_bytes_backward": 65535,
            "act_data_pkt_fwd": 5,
            "min_seg_size_forward": 0,
            "Active Mean": 100,
            "Active Std": 20,
            "Active Max": 200,
            "Active Min": 50,
            "Idle Mean": 0,
            "Idle Std": 0,
            "Idle Max": 0,
            "Idle Min": 0
        },
        "source_ip": f"192.168.1.{random.randint(50, 99)}",
        "attack_type": "Benign"
    }

def main():
    """Run live threat demo"""
    print("🚀 LIVE THREAT DEMO - AI CYBERSECURITY TOOL")
    print("=" * 50)
    print("📊 Watch your dashboard at http://localhost:3000")
    print("🔄 This will send threats every 3-5 seconds")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 50)
    
    attack_count = 0
    
    try:
        while True:
            attack_count += 1
            
            # Randomly choose attack type (70% attacks, 30% benign)
            if random.random() < 0.7:
                attack_data = create_ddos_attack()
                attack_type = "DDoS Attack"
            else:
                attack_data = create_benign_traffic()
                attack_type = "Benign Traffic"
            
            print(f"\n🎯 Attack #{attack_count} - {datetime.now().strftime('%H:%M:%S')}")
            result = send_threat(attack_type, attack_data)
            
            if result and result.get('threat_detected'):
                print("🚨 THREAT DETECTED! Check dashboard for alerts!")
            
            # Wait 3-5 seconds before next attack
            wait_time = random.randint(3, 5)
            print(f"⏳ Next attack in {wait_time} seconds...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Demo stopped after {attack_count} attacks")
        print("📊 Check your dashboard for final statistics!")

if __name__ == "__main__":
    main()
