#!/usr/bin/env python3
"""
Real Threat Simulation for AI Cybersecurity Tool
Simulates realistic attack patterns to test threat detection
"""

import requests
import json
import time
import random
from datetime import datetime
import threading

# API Configuration
API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"

class ThreatSimulator:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY
        })
        
    def send_threat_data(self, threat_data, attack_name):
        """Send threat data to API and return response"""
        try:
            response = self.session.post(f"{API_BASE_URL}/predict", json=threat_data)
            if response.status_code == 200:
                data = response.json()
                threat_score = data.get('threat_score', 0)
                threat_level = data.get('threat_level', 'Unknown')
                processing_time = data.get('processing_time_ms', 0)
                
                print(f"🚨 {attack_name}")
                print(f"   🎯 Threat Score: {threat_score:.3f} ({threat_score*100:.1f}%)")
                print(f"   🚨 Threat Level: {threat_level}")
                print(f"   ⚡ Processing Time: {processing_time:.1f}ms")
                print(f"   🤖 Model Predictions: {data.get('model_predictions', {})}")
                print(f"   🔍 Threat Detected: {'YES' if data.get('threat_detected') else 'NO'}")
                print()
                
                return data
            else:
                print(f"❌ {attack_name} failed: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ {attack_name} error: {e}")
            return None

    def simulate_ddos_attack(self):
        """Simulate a DDoS attack pattern"""
        print("🔥 SIMULATING DDoS ATTACK")
        print("=" * 50)
        
        # DDoS characteristics: High packet rate, short duration, many connections
        ddos_data = {
            "features": {
                "Destination Port": 80,
                "Flow Duration": 50,  # Very short duration
                "Total Fwd Packets": 10000,  # Extremely high packet count
                "Total Backward Packets": 100,  # Very few responses
                "Total Length of Fwd Packets": 1000000,  # Large data volume
                "Total Length of Bwd Packets": 10000,  # Small response
                "Fwd Packet Length Max": 1500,
                "Fwd Packet Length Min": 64,
                "Fwd Packet Length Mean": 100,
                "Fwd Packet Length Std": 200,
                "Bwd Packet Length Max": 100,
                "Bwd Packet Length Min": 0,
                "Bwd Packet Length Mean": 10,
                "Bwd Packet Length Std": 20,
                "Flow Bytes/s": 20000000,  # Extremely high byte rate
                "Flow Packets/s": 200000,  # Extremely high packet rate
                "Flow IAT Mean": 0.5,  # Very short intervals
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
                "SYN Flag Count": 10000,  # Many SYN packets
                "RST Flag Count": 0,
                "PSH Flag Count": 0,
                "ACK Flag Count": 100,
                "URG Flag Count": 0,
                "CWE Flag Count": 0,
                "ECE Flag Count": 0,
                "Down/Up Ratio": 0.01,  # Very low response ratio
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
            },
            "source_ip": "192.168.1.100",
            "attack_type": "DDoS"
        }
        
        return self.send_threat_data(ddos_data, "DDoS Attack")

    def simulate_port_scan(self):
        """Simulate a port scan attack"""
        print("🔍 SIMULATING PORT SCAN ATTACK")
        print("=" * 50)
        
        # Port scan characteristics: Many different ports, short connections
        port_scan_data = {
            "features": {
                "Destination Port": random.randint(1, 65535),  # Random high port
                "Flow Duration": 100,  # Short duration
                "Total Fwd Packets": 1,  # Single packet
                "Total Backward Packets": 1,  # Single response
                "Total Length of Fwd Packets": 60,  # Small packet
                "Total Length of Bwd Packets": 60,  # Small response
                "Fwd Packet Length Max": 60,
                "Fwd Packet Length Min": 60,
                "Fwd Packet Length Mean": 60,
                "Fwd Packet Length Std": 0,
                "Bwd Packet Length Max": 60,
                "Bwd Packet Length Min": 60,
                "Bwd Packet Length Mean": 60,
                "Bwd Packet Length Std": 0,
                "Flow Bytes/s": 600,  # Low byte rate
                "Flow Packets/s": 10,  # Low packet rate
                "Flow IAT Mean": 100,
                "Flow IAT Std": 0,
                "Flow IAT Max": 100,
                "Flow IAT Min": 100,
                "Fwd IAT Total": 100,
                "Fwd IAT Mean": 100,
                "Fwd IAT Std": 0,
                "Fwd IAT Max": 100,
                "Fwd IAT Min": 100,
                "Bwd IAT Total": 100,
                "Bwd IAT Mean": 100,
                "Bwd IAT Std": 0,
                "Bwd IAT Max": 100,
                "Bwd IAT Min": 100,
                "Fwd PSH Flags": 0,
                "Bwd PSH Flags": 0,
                "Fwd URG Flags": 0,
                "Bwd URG Flags": 0,
                "Fwd Header Length": 20,
                "Bwd Header Length": 20,
                "Fwd Packets/s": 10,
                "Bwd Packets/s": 10,
                "Min Packet Length": 60,
                "Max Packet Length": 60,
                "Packet Length Mean": 60,
                "Packet Length Std": 0,
                "Packet Length Variance": 0,
                "FIN Flag Count": 0,
                "SYN Flag Count": 1,  # SYN scan
                "RST Flag Count": 0,
                "PSH Flag Count": 0,
                "ACK Flag Count": 0,
                "URG Flag Count": 0,
                "CWE Flag Count": 0,
                "ECE Flag Count": 0,
                "Down/Up Ratio": 1.0,  # Equal response
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
            },
            "source_ip": "10.0.0.50",
            "attack_type": "Port Scan"
        }
        
        return self.send_threat_data(port_scan_data, "Port Scan Attack")

    def simulate_brute_force(self):
        """Simulate a brute force attack"""
        print("💥 SIMULATING BRUTE FORCE ATTACK")
        print("=" * 50)
        
        # Brute force characteristics: Many failed attempts, consistent patterns
        brute_force_data = {
            "features": {
                "Destination Port": 22,  # SSH port
                "Flow Duration": 5000,  # Longer duration
                "Total Fwd Packets": 100,  # Many attempts
                "Total Backward Packets": 100,  # Many responses
                "Total Length of Fwd Packets": 10000,  # Login attempts
                "Total Length of Bwd Packets": 5000,  # Failed responses
                "Fwd Packet Length Max": 200,
                "Fwd Packet Length Min": 100,
                "Fwd Packet Length Mean": 150,
                "Fwd Packet Length Std": 30,
                "Bwd Packet Length Max": 100,
                "Bwd Packet Length Min": 50,
                "Bwd Packet Length Mean": 75,
                "Bwd Packet Length Std": 15,
                "Flow Bytes/s": 2000,  # Moderate byte rate
                "Flow Packets/s": 20,  # Moderate packet rate
                "Flow IAT Mean": 50,  # Regular intervals
                "Flow IAT Std": 10,
                "Flow IAT Max": 100,
                "Flow IAT Min": 30,
                "Fwd IAT Total": 5000,
                "Fwd IAT Mean": 50,
                "Fwd IAT Std": 10,
                "Fwd IAT Max": 100,
                "Fwd IAT Min": 30,
                "Bwd IAT Total": 5000,
                "Bwd IAT Mean": 50,
                "Bwd IAT Std": 10,
                "Bwd IAT Max": 100,
                "Bwd IAT Min": 30,
                "Fwd PSH Flags": 100,  # Many PSH flags (data packets)
                "Bwd PSH Flags": 100,  # Many PSH flags (responses)
                "Fwd URG Flags": 0,
                "Bwd URG Flags": 0,
                "Fwd Header Length": 20,
                "Bwd Header Length": 20,
                "Fwd Packets/s": 20,
                "Bwd Packets/s": 20,
                "Min Packet Length": 50,
                "Max Packet Length": 200,
                "Packet Length Mean": 125,
                "Packet Length Std": 50,
                "Packet Length Variance": 2500,
                "FIN Flag Count": 0,
                "SYN Flag Count": 1,
                "RST Flag Count": 0,
                "PSH Flag Count": 200,  # Many data packets
                "ACK Flag Count": 200,  # Many acknowledgments
                "URG Flag Count": 0,
                "CWE Flag Count": 0,
                "ECE Flag Count": 0,
                "Down/Up Ratio": 0.5,  # Half response rate (failed attempts)
                "Average Packet Size": 125,
                "Avg Fwd Segment Size": 150,
                "Avg Bwd Segment Size": 75,
                "Fwd Header Length.1": 20,
                "Fwd Avg Bytes/Bulk": 0,
                "Fwd Avg Packets/Bulk": 0,
                "Fwd Avg Bulk Rate": 0,
                "Bwd Avg Bytes/Bulk": 0,
                "Bwd Avg Packets/Bulk": 0,
                "Bwd Avg Bulk Rate": 0,
                "Subflow Fwd Packets": 100,
                "Subflow Fwd Bytes": 10000,
                "Subflow Bwd Packets": 100,
                "Subflow Bwd Bytes": 5000,
                "Init_Win_bytes_forward": 65535,
                "Init_Win_bytes_backward": 65535,
                "act_data_pkt_fwd": 100,  # Many data packets
                "min_seg_size_forward": 0,
                "Active Mean": 50,
                "Active Std": 10,
                "Active Max": 100,
                "Active Min": 30,
                "Idle Mean": 0,
                "Idle Std": 0,
                "Idle Max": 0,
                "Idle Min": 0
            },
            "source_ip": "172.16.0.25",
            "attack_type": "Brute Force"
        }
        
        return self.send_threat_data(brute_force_data, "Brute Force Attack")

    def simulate_benign_traffic(self):
        """Simulate normal, benign traffic"""
        print("✅ SIMULATING BENIGN TRAFFIC")
        print("=" * 50)
        
        # Normal traffic characteristics: balanced, regular patterns
        benign_data = {
            "features": {
                "Destination Port": 80,  # HTTP port
                "Flow Duration": 2000,  # Normal duration
                "Total Fwd Packets": 20,  # Normal packet count
                "Total Backward Packets": 18,  # Normal response count
                "Total Length of Fwd Packets": 2000,  # Normal data volume
                "Total Length of Bwd Packets": 1800,  # Normal response volume
                "Fwd Packet Length Max": 1500,
                "Fwd Packet Length Min": 64,
                "Fwd Packet Length Mean": 100,
                "Fwd Packet Length Std": 200,
                "Bwd Packet Length Max": 1500,
                "Bwd Packet Length Min": 64,
                "Bwd Packet Length Mean": 100,
                "Bwd Packet Length Std": 200,
                "Flow Bytes/s": 1000,  # Normal byte rate
                "Flow Packets/s": 10,  # Normal packet rate
                "Flow IAT Mean": 100,  # Normal intervals
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
                "Fwd PSH Flags": 5,  # Some data packets
                "Bwd PSH Flags": 5,  # Some data packets
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
                "FIN Flag Count": 1,  # Proper connection close
                "SYN Flag Count": 1,  # Connection establishment
                "RST Flag Count": 0,  # No resets
                "PSH Flag Count": 10,  # Some data packets
                "ACK Flag Count": 20,  # Proper acknowledgments
                "URG Flag Count": 0,
                "CWE Flag Count": 0,
                "ECE Flag Count": 0,
                "Down/Up Ratio": 0.9,  # Good response ratio
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
                "act_data_pkt_fwd": 5,  # Some data packets
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
            "source_ip": "192.168.1.50",
            "attack_type": "Benign"
        }
        
        return self.send_threat_data(benign_data, "Benign Traffic")

    def run_continuous_simulation(self, duration_minutes=5):
        """Run continuous threat simulation"""
        print(f"🚀 STARTING CONTINUOUS THREAT SIMULATION")
        print(f"⏱️ Duration: {duration_minutes} minutes")
        print(f"🔄 Watch your dashboard for real-time updates!")
        print("=" * 60)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        attack_types = [
            self.simulate_benign_traffic,
            self.simulate_ddos_attack,
            self.simulate_port_scan,
            self.simulate_brute_force
        ]
        
        attack_names = ["Benign", "DDoS", "Port Scan", "Brute Force"]
        
        while time.time() < end_time:
            # Randomly select an attack type
            attack_idx = random.randint(0, len(attack_types) - 1)
            attack_func = attack_types[attack_idx]
            attack_name = attack_names[attack_idx]
            
            print(f"\n🎲 Random Attack: {attack_name}")
            print(f"⏰ Time remaining: {int((end_time - time.time()) / 60)} minutes")
            
            # Execute the attack
            result = attack_func()
            
            if result and result.get('threat_detected'):
                print(f"🚨 THREAT DETECTED! Check your dashboard!")
            
            # Wait before next attack
            wait_time = random.randint(10, 30)  # 10-30 seconds between attacks
            print(f"⏳ Waiting {wait_time} seconds before next attack...")
            time.sleep(wait_time)
        
        print(f"\n✅ Simulation completed!")
        print(f"📊 Check your dashboard for final statistics")

def main():
    """Main function to run threat simulations"""
    print("🛡️ AI CYBERSECURITY TOOL - REAL THREAT SIMULATION")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API URL: {API_BASE_URL}")
    print(f"🔑 API Key: {API_KEY}")
    print()
    
    simulator = ThreatSimulator()
    
    # Test API connection first
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API connection successful")
        else:
            print("❌ API connection failed")
            return
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return
    
    print("\n🎯 Choose simulation mode:")
    print("1. Single DDoS Attack Test")
    print("2. Single Port Scan Test") 
    print("3. Single Brute Force Test")
    print("4. Single Benign Traffic Test")
    print("5. Continuous Mixed Simulation (5 minutes)")
    print("6. Continuous Mixed Simulation (10 minutes)")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            simulator.simulate_ddos_attack()
        elif choice == "2":
            simulator.simulate_port_scan()
        elif choice == "3":
            simulator.simulate_brute_force()
        elif choice == "4":
            simulator.simulate_benign_traffic()
        elif choice == "5":
            simulator.run_continuous_simulation(5)
        elif choice == "6":
            simulator.run_continuous_simulation(10)
        else:
            print("Invalid choice. Running DDoS test...")
            simulator.simulate_ddos_attack()
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n🎉 Simulation complete!")
    print("📊 Check your dashboard at http://localhost:3000")
    print("📈 Watch the real-time threat scores and alerts!")

if __name__ == "__main__":
    main()
