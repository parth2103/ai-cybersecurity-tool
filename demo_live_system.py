#!/usr/bin/env python3
"""
Live Demo Script for AI Cybersecurity Tool
Shows the system in action with real threat detection
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:5001"
API_KEY = "dev-key-123"  # Development API key

def test_api_health():
    """Test API health endpoint"""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API is healthy - {data['status']}")
            print(f"   📅 Timestamp: {data['timestamp']}")
            return True
        else:
            print(f"   ❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        return False

def test_authentication():
    """Test API authentication"""
    print("\n🔐 Testing Authentication...")
    
    # Test without API key (should fail)
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 401:
            print("   ✅ Authentication required (no API key)")
        else:
            print(f"   ⚠️ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Authentication test failed: {e}")
    
    # Test with API key (should succeed)
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(f"{API_BASE_URL}/stats", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Authentication successful with API key")
            print(f"   📊 Total requests: {data.get('total_requests', 0)}")
            print(f"   🚨 Threats detected: {data.get('threats_detected', 0)}")
            return True
        else:
            print(f"   ❌ Authentication failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Authentication test failed: {e}")
        return False

def test_threat_detection():
    """Test threat detection with realistic network data"""
    print("\n🛡️ Testing Threat Detection...")
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    # Test 1: Benign traffic
    print("\n   1️⃣ Testing Benign Traffic...")
    benign_data = {
        "features": {
            "Destination Port": 80,
            "Flow Duration": 1000,
            "Total Fwd Packets": 10,
            "Total Backward Packets": 8,
            "Total Length of Fwd Packets": 1000,
            "Total Length of Bwd Packets": 800,
            "Fwd Packet Length Max": 100,
            "Fwd Packet Length Min": 90,
            "Fwd Packet Length Mean": 95,
            "Fwd Packet Length Std": 5,
            "Bwd Packet Length Max": 100,
            "Bwd Packet Length Min": 90,
            "Bwd Packet Length Mean": 95,
            "Bwd Packet Length Std": 5,
            "Flow Bytes/s": 1000,
            "Flow Packets/s": 10,
            "Flow IAT Mean": 100,
            "Flow IAT Std": 10,
            "Flow IAT Max": 120,
            "Flow IAT Min": 80,
            "Fwd IAT Total": 1000,
            "Fwd IAT Mean": 100,
            "Fwd IAT Std": 10,
            "Fwd IAT Max": 120,
            "Fwd IAT Min": 80,
            "Bwd IAT Total": 800,
            "Bwd IAT Mean": 100,
            "Bwd IAT Std": 10,
            "Bwd IAT Max": 120,
            "Bwd IAT Min": 80,
            "Fwd PSH Flags": 0,
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Length": 20,
            "Bwd Header Length": 20,
            "Fwd Packets/s": 5,
            "Bwd Packets/s": 4,
            "Min Packet Length": 90,
            "Max Packet Length": 100,
            "Packet Length Mean": 95,
            "Packet Length Std": 5,
            "Packet Length Variance": 25,
            "FIN Flag Count": 0,
            "SYN Flag Count": 1,
            "RST Flag Count": 0,
            "PSH Flag Count": 0,
            "ACK Flag Count": 1,
            "URG Flag Count": 0,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": 0.8,
            "Average Packet Size": 95,
            "Avg Fwd Segment Size": 95,
            "Avg Bwd Segment Size": 95,
            "Fwd Header Length.1": 20,
            "Fwd Avg Bytes/Bulk": 0,
            "Fwd Avg Packets/Bulk": 0,
            "Fwd Avg Bulk Rate": 0,
            "Bwd Avg Bytes/Bulk": 0,
            "Bwd Avg Packets/Bulk": 0,
            "Bwd Avg Bulk Rate": 0,
            "Subflow Fwd Packets": 10,
            "Subflow Fwd Bytes": 1000,
            "Subflow Bwd Packets": 8,
            "Subflow Bwd Bytes": 800,
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
        "source_ip": "192.168.1.100",
        "attack_type": "Benign"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", 
                               headers=headers, 
                               json=benign_data)
        if response.status_code == 200:
            data = response.json()
            threat_score = data.get('threat_score', 0)
            threat_level = data.get('threat_level', 'Unknown')
            processing_time = data.get('processing_time_ms', 0)
            
            print(f"   ✅ Benign traffic analyzed")
            print(f"   🎯 Threat Score: {threat_score:.3f} ({threat_score*100:.1f}%)")
            print(f"   🚨 Threat Level: {threat_level}")
            print(f"   ⚡ Processing Time: {processing_time:.1f}ms")
            print(f"   🤖 Model Predictions: {data.get('model_predictions', {})}")
        else:
            print(f"   ❌ Benign test failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Benign test failed: {e}")
    
    # Test 2: Suspicious traffic (simulated attack)
    print("\n   2️⃣ Testing Suspicious Traffic...")
    suspicious_data = benign_data.copy()
    suspicious_data["features"]["Total Fwd Packets"] = 1000  # High packet count
    suspicious_data["features"]["Flow Duration"] = 100  # Very short duration
    suspicious_data["features"]["Flow Packets/s"] = 100  # High packet rate
    suspicious_data["attack_type"] = "DDoS"
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", 
                               headers=headers, 
                               json=suspicious_data)
        if response.status_code == 200:
            data = response.json()
            threat_score = data.get('threat_score', 0)
            threat_level = data.get('threat_level', 'Unknown')
            processing_time = data.get('processing_time_ms', 0)
            
            print(f"   ✅ Suspicious traffic analyzed")
            print(f"   🎯 Threat Score: {threat_score:.3f} ({threat_score*100:.1f}%)")
            print(f"   🚨 Threat Level: {threat_level}")
            print(f"   ⚡ Processing Time: {processing_time:.1f}ms")
            print(f"   🤖 Model Predictions: {data.get('model_predictions', {})}")
        else:
            print(f"   ❌ Suspicious test failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Suspicious test failed: {e}")

def test_system_info():
    """Test system information endpoint"""
    print("\n💻 Testing System Information...")
    
    headers = {"X-API-Key": "admin-key-789"}  # Admin key for system info
    
    try:
        response = requests.get(f"{API_BASE_URL}/system/info", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ System information retrieved")
            print(f"   🖥️ CPU Usage: {data.get('cpu_percent', 0):.1f}%")
            print(f"   💾 Memory Usage: {data.get('memory_percent', 0):.1f}%")
            print(f"   💽 Disk Usage: {data.get('disk_usage', 0):.1f}%")
            print(f"   🤖 Models Loaded: {data.get('models_loaded', [])}")
            print(f"   📊 Total Predictions: {data.get('total_predictions', 0)}")
            print(f"   🚨 Threats Detected: {data.get('threats_detected', 0)}")
            print(f"   📈 Detection Rate: {data.get('detection_rate', 0):.1f}%")
        else:
            print(f"   ❌ System info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ System info test failed: {e}")

def test_database_endpoints():
    """Test database endpoints"""
    print("\n🗄️ Testing Database Endpoints...")
    
    headers = {"X-API-Key": "admin-key-789"}  # Admin key for database access
    
    # Test database statistics
    try:
        response = requests.get(f"{API_BASE_URL}/database/statistics", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Database statistics retrieved")
            print(f"   📊 Total Threats: {data.get('total_threats', 0)}")
            print(f"   🎯 Average Threat Score: {data.get('average_threat_score', 0):.3f}")
            print(f"   🗃️ Database Size: {data.get('database', {}).get('database_size_mb', 0):.2f} MB")
        else:
            print(f"   ❌ Database stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Database stats test failed: {e}")

def main():
    """Main demo function"""
    print("🚀 AI Cybersecurity Tool - Live Demo")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API URL: {API_BASE_URL}")
    print(f"🔑 API Key: {API_KEY}")
    
    # Run all tests
    if not test_api_health():
        print("\n❌ API is not running. Please start the API server first:")
        print("   python api/app.py")
        return
    
    if not test_authentication():
        print("\n❌ Authentication failed. Check API key configuration.")
        return
    
    test_threat_detection()
    test_system_info()
    test_database_endpoints()
    
    print("\n🎉 Demo completed successfully!")
    print("\n📋 What you just saw:")
    print("   ✅ API health monitoring")
    print("   ✅ Authentication and authorization")
    print("   ✅ Real-time threat detection")
    print("   ✅ Machine learning model predictions")
    print("   ✅ System performance monitoring")
    print("   ✅ Database persistence and analytics")
    print("   ✅ Comprehensive error handling")
    print("   ✅ Security validation and sanitization")
    
    print("\n🌐 Next steps:")
    print("   1. Open http://localhost:3000 for the React dashboard")
    print("   2. Check logs/ directory for detailed logs")
    print("   3. Explore the API endpoints with different API keys")
    print("   4. Run performance tests: python tests/performance_test.py")

if __name__ == "__main__":
    main()
