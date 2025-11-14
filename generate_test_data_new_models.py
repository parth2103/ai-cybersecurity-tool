#!/usr/bin/env python3
"""
Generate test data for new models
Creates benign and attack patterns for IoT/IIoT traffic
"""

import numpy as np
import joblib
import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "test_data"


def load_feature_names():
    """Load new model feature names"""
    features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
    if features_path.exists():
        features = joblib.load(features_path)
        if not isinstance(features, list):
            features = list(features)
        return features
    return None


def generate_benign_traffic(n_samples=100):
    """Generate benign IoT/IIoT traffic patterns"""
    feature_names = load_feature_names()
    if feature_names is None:
        return []
    
    samples = []
    for i in range(n_samples):
        features = {}
        for feat in feature_names:
            if "Port" in feat:
                features[feat] = np.random.choice([443, 80, 22, 53])  # HTTPS, HTTP, SSH, DNS
            elif "Duration" in feat:
                features[feat] = np.random.randint(50000, 500000)  # Normal duration
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                features[feat] = np.random.randint(5, 50)  # Normal packet count
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                features[feat] = np.random.randint(3, 40)  # Normal response
            elif "Bytes/s" in feat or "Bytes per" in feat:
                features[feat] = np.random.uniform(1000, 10000)  # Normal throughput
            elif "Packets/s" in feat or "Packets per" in feat:
                features[feat] = np.random.uniform(10, 100)  # Normal rate
            elif "IAT" in feat or "Inter Arrival" in feat:
                features[feat] = np.random.uniform(100, 1000)  # Normal intervals
            else:
                features[feat] = np.random.uniform(0, 100)
        
        samples.append({
            "features": features,
            "label": "benign",
            "attack_type": "normal_traffic"
        })
    
    return samples


def generate_dos_attack(n_samples=50):
    """Generate DoS attack patterns"""
    feature_names = load_feature_names()
    if feature_names is None:
        return []
    
    samples = []
    for i in range(n_samples):
        features = {}
        for feat in feature_names:
            if "Port" in feat:
                features[feat] = 80  # HTTP
            elif "Duration" in feat:
                features[feat] = np.random.randint(500, 5000)  # Short duration
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                features[feat] = np.random.randint(10000, 100000)  # High packet count
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                features[feat] = 0  # No response
            elif "Bytes/s" in feat or "Bytes per" in feat:
                features[feat] = np.random.uniform(10000000, 100000000)  # High throughput
            elif "Packets/s" in feat or "Packets per" in feat:
                features[feat] = np.random.uniform(10000, 100000)  # High rate
            elif "IAT" in feat or "Inter Arrival" in feat:
                features[feat] = np.random.uniform(0.01, 0.1)  # Very short intervals
            else:
                features[feat] = np.random.uniform(0, 1000)
        
        samples.append({
            "features": features,
            "label": "attack",
            "attack_type": "dos"
        })
    
    return samples


def generate_ddos_attack(n_samples=50):
    """Generate DDoS attack patterns"""
    feature_names = load_feature_names()
    if feature_names is None:
        return []
    
    samples = []
    for i in range(n_samples):
        features = {}
        for feat in feature_names:
            if "Port" in feat:
                features[feat] = 80
            elif "Duration" in feat:
                features[feat] = np.random.randint(100, 1000)  # Very short
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                features[feat] = np.random.randint(100000, 1000000)  # Massive count
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                features[feat] = 0
            elif "Bytes/s" in feat or "Bytes per" in feat:
                features[feat] = np.random.uniform(100000000, 1000000000)  # Very high
            elif "Packets/s" in feat or "Packets per" in feat:
                features[feat] = np.random.uniform(100000, 1000000)  # Very high rate
            elif "IAT" in feat or "Inter Arrival" in feat:
                features[feat] = np.random.uniform(0.001, 0.01)  # Extremely short
            else:
                features[feat] = np.random.uniform(0, 2000)
        
        samples.append({
            "features": features,
            "label": "attack",
            "attack_type": "ddos"
        })
    
    return samples


def generate_mirai_botnet(n_samples=50):
    """Generate Mirai botnet attack patterns"""
    feature_names = load_feature_names()
    if feature_names is None:
        return []
    
    samples = []
    for i in range(n_samples):
        features = {}
        for feat in feature_names:
            if "Port" in feat:
                features[feat] = 23  # Telnet
            elif "Duration" in feat:
                features[feat] = np.random.randint(10000, 100000)  # Medium
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                features[feat] = np.random.randint(1000, 10000)  # Steady stream
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                features[feat] = np.random.randint(50, 500)  # Some response
            elif "Bytes/s" in feat or "Bytes per" in feat:
                features[feat] = np.random.uniform(1000000, 10000000)  # Moderate
            elif "Packets/s" in feat or "Packets per" in feat:
                features[feat] = np.random.uniform(100, 500)  # Steady rate
            elif "IAT" in feat or "Inter Arrival" in feat:
                features[feat] = np.random.uniform(2, 10)  # Regular intervals
            else:
                features[feat] = np.random.uniform(0, 500)
        
        samples.append({
            "features": features,
            "label": "attack",
            "attack_type": "mirai_botnet"
        })
    
    return samples


def generate_brute_force(n_samples=50):
    """Generate Brute Force attack patterns"""
    feature_names = load_feature_names()
    if feature_names is None:
        return []
    
    samples = []
    for i in range(n_samples):
        features = {}
        for feat in feature_names:
            if "Port" in feat:
                features[feat] = 22  # SSH
            elif "Duration" in feat:
                features[feat] = np.random.randint(500, 5000)  # Short attempts
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                features[feat] = np.random.randint(10, 100)  # Multiple attempts
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                features[feat] = np.random.randint(10, 100)  # Responses
            elif "Bytes/s" in feat or "Bytes per" in feat:
                features[feat] = np.random.uniform(1000, 10000)  # Low throughput
            elif "Packets/s" in feat or "Packets per" in feat:
                features[feat] = np.random.uniform(20, 200)  # Moderate rate
            elif "IAT" in feat or "Inter Arrival" in feat:
                features[feat] = np.random.uniform(5, 50)  # Regular intervals
            else:
                features[feat] = np.random.uniform(0, 100)
        
        samples.append({
            "features": features,
            "label": "attack",
            "attack_type": "brute_force"
        })
    
    return samples


def generate_recon(n_samples=50):
    """Generate Recon attack patterns"""
    feature_names = load_feature_names()
    if feature_names is None:
        return []
    
    samples = []
    for i in range(n_samples):
        features = {}
        for feat in feature_names:
            if "Port" in feat:
                features[feat] = np.random.choice([22, 80, 443, 3389])  # Various ports
            elif "Duration" in feat:
                features[feat] = np.random.randint(50, 500)  # Very short
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                features[feat] = 1  # Single packet
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                features[feat] = 0  # No response (port closed)
            elif "Bytes/s" in feat or "Bytes per" in feat:
                features[feat] = np.random.uniform(50, 200)  # Very low
            elif "Packets/s" in feat or "Packets per" in feat:
                features[feat] = np.random.uniform(1, 20)  # Low rate
            elif "IAT" in feat or "Inter Arrival" in feat:
                features[feat] = np.random.uniform(50, 500)  # Longer intervals
            else:
                features[feat] = np.random.uniform(0, 50)
        
        samples.append({
            "features": features,
            "label": "attack",
            "attack_type": "recon"
        })
    
    return samples


def generate_spoofing(n_samples=50):
    """Generate Spoofing attack patterns"""
    feature_names = load_feature_names()
    if feature_names is None:
        return []
    
    samples = []
    for i in range(n_samples):
        features = {}
        for feat in feature_names:
            if "Port" in feat:
                features[feat] = 53  # DNS
            elif "Duration" in feat:
                features[feat] = np.random.randint(1000, 10000)  # Medium
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                features[feat] = np.random.randint(100, 1000)  # Moderate
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                features[feat] = 0  # No legitimate response
            elif "Bytes/s" in feat or "Bytes per" in feat:
                features[feat] = np.random.uniform(10000, 100000)  # Moderate
            elif "Packets/s" in feat or "Packets per" in feat:
                features[feat] = np.random.uniform(50, 500)  # Moderate rate
            elif "IAT" in feat or "Inter Arrival" in feat:
                features[feat] = np.random.uniform(2, 20)  # Regular but suspicious
            else:
                features[feat] = np.random.uniform(0, 200)
        
        samples.append({
            "features": features,
            "label": "attack",
            "attack_type": "spoofing"
        })
    
    return samples


def generate_iiot_attack(n_samples=50):
    """Generate IIoT attack patterns (CICAPT-IIOT specific)"""
    feature_names = load_feature_names()
    if feature_names is None:
        return []
    
    samples = []
    for i in range(n_samples):
        features = {}
        for feat in feature_names:
            if "Port" in feat:
                features[feat] = 502  # Modbus
            elif "Duration" in feat:
                features[feat] = np.random.randint(5000, 50000)  # Longer
            elif "Fwd Packets" in feat or "Forward Packets" in feat:
                features[feat] = np.random.randint(1000, 10000)  # Command packets
            elif "Bwd Packets" in feat or "Backward Packets" in feat:
                features[feat] = np.random.randint(1000, 10000)  # Responses
            elif "Bytes/s" in feat or "Bytes per" in feat:
                features[feat] = np.random.uniform(5000000, 50000000)  # High
            elif "Packets/s" in feat or "Packets per" in feat:
                features[feat] = np.random.uniform(50, 500)  # Moderate rate
            elif "IAT" in feat or "Inter Arrival" in feat:
                features[feat] = np.random.uniform(2, 20)  # Regular intervals
            else:
                features[feat] = np.random.uniform(0, 300)
        
        samples.append({
            "features": features,
            "label": "attack",
            "attack_type": "iiot_attack"
        })
    
    return samples


def make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    else:
        return obj


def main():
    """Generate all test data"""
    print("=" * 60)
    print("GENERATING TEST DATA FOR NEW MODELS")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Generate all patterns
    print("\nGenerating test patterns...")
    
    all_samples = []
    
    print("  - Benign traffic...")
    all_samples.extend(generate_benign_traffic(100))
    
    print("  - DoS attacks...")
    all_samples.extend(generate_dos_attack(50))
    
    print("  - DDoS attacks...")
    all_samples.extend(generate_ddos_attack(50))
    
    print("  - Mirai botnet...")
    all_samples.extend(generate_mirai_botnet(50))
    
    print("  - Brute Force...")
    all_samples.extend(generate_brute_force(50))
    
    print("  - Recon...")
    all_samples.extend(generate_recon(50))
    
    print("  - Spoofing...")
    all_samples.extend(generate_spoofing(50))
    
    print("  - IIoT attacks...")
    all_samples.extend(generate_iiot_attack(50))
    
    # Convert numpy types to Python native types
    print("\nConverting to JSON-serializable format...")
    all_samples = make_serializable(all_samples)
    
    # Save to JSON
    output_file = OUTPUT_DIR / "test_data_new_models.json"
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"\n✅ Generated {len(all_samples)} test samples")
    print(f"✅ Saved to: {output_file}")
    
    # Summary
    attack_types = {}
    for sample in all_samples:
        attack_type = sample.get("attack_type", "unknown")
        attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
    
    print("\nTest Data Summary:")
    for attack_type, count in sorted(attack_types.items()):
        print(f"  {attack_type}: {count} samples")
    
    print("\n✅ Test data generation complete!")


if __name__ == "__main__":
    main()

