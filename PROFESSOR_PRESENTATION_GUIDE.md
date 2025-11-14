# AI Cybersecurity Tool - Professor Presentation Guide

## 🎯 Overview
This document provides evidence and details about the AI Cybersecurity Tool's use of **new IoT/IIoT datasets** and the **threat types** simulated for testing.

---

## 📊 **Evidence: New Datasets Are Being Used**

### 1. **Model Files (Proof of New Dataset Training)**
The system uses models trained on **new datasets** (IoT-IDAD 2024 + CICAPT-IIOT):

```
models/
├── random_forest_new_datasets.pkl      ✅ NEW (IoT-IDAD + CICAPT)
├── xgboost_model_new_datasets.pkl      ✅ NEW (IoT-IDAD + CICAPT)
├── isolation_forest_new_datasets.pkl    ✅ NEW (IoT-IDAD + CICAPT)
├── scaler_new_datasets.pkl              ✅ NEW (IoT-IDAD + CICAPT)
└── feature_names_new_datasets.pkl       ✅ NEW (145 features vs 78 old)
```

**Verification Command:**
```bash
ls -lh models/*new_datasets.pkl
```

### 2. **API Logs Show New Models Loading**
Check the API startup logs:
```bash
tail -20 logs/api_startup.log | grep -i "new\|iot\|cicapt"
```

You'll see entries like:
```
Loaded RF model from .../random_forest_new_datasets.pkl [NEW (IoT-IDAD + CICAPT)]
Loaded XGBoost model from .../xgboost_model_new_datasets.pkl [NEW (IoT-IDAD + CICAPT)]
Loaded Isolation Forest model from .../isolation_forest_new_datasets.pkl [NEW (IoT-IDAD + CICAPT)]
Loaded feature list from .../feature_names_new_datasets.pkl [NEW]
```

### 3. **Feature Count Difference**
- **Old Models (CICIDS2017):** 78 features
- **New Models (IoT-IDAD + CICAPT):** 145 features

**Verification:**
```bash
python3 -c "import joblib; print('Old:', len(joblib.load('models/feature_names.pkl'))); print('New:', len(joblib.load('models/feature_names_new_datasets.pkl')))"
```

### 4. **API System Info Endpoint**
Check which models are loaded:
```bash
curl -H "X-API-Key: dev-key-123" http://localhost:5001/system/info | python3 -m json.tool
```

Look for `models_loaded` array - it should contain `*_new_datasets.pkl` files.

### 5. **Training Results**
Check the training results:
```bash
cat results/training_results_new_datasets_*.json
```

This shows:
- Datasets used: `cic_iot_2024`, `cicapt_iiot`
- Number of samples trained on
- Model performance metrics

---

## 🎭 **Threat Types Simulated in Test Data**

The test data (`test_data/test_data_new_models.json`) includes **8 different threat types**:

### 1. **Normal Traffic (Benign)**
- **Purpose:** Baseline legitimate IoT/IIoT traffic
- **Characteristics:**
  - Ports: 443 (HTTPS), 80 (HTTP), 22 (SSH), 53 (DNS)
  - Duration: 50,000-500,000 ms (normal session length)
  - Packet counts: 5-50 forward, 3-40 backward
  - Throughput: 1,000-10,000 bytes/s
  - **Samples:** 100

### 2. **DoS (Denial of Service)**
- **Purpose:** Single-source service disruption
- **Characteristics:**
  - Port: 80 (HTTP)
  - Duration: 500-5,000 ms (short-lived)
  - Packet counts: 10,000-100,000 forward, 0 backward (no response)
  - Throughput: 10M-100M bytes/s (high volume)
  - Inter-arrival time: 0.01-0.1 ms (very fast)
  - **Samples:** 50

### 3. **DDoS (Distributed Denial of Service)**
- **Purpose:** Multi-source coordinated attack
- **Characteristics:**
  - Port: 80
  - Duration: 100-1,000 ms (extremely short)
  - Packet counts: 100,000-1,000,000 forward (massive)
  - Throughput: 100M-1B bytes/s (extremely high)
  - Inter-arrival time: 0.001-0.01 ms (extremely fast)
  - **Samples:** 50

### 4. **Mirai Botnet**
- **Purpose:** IoT device hijacking and botnet formation
- **Characteristics:**
  - Port: 23 (Telnet - Mirai's primary target)
  - Duration: 10,000-100,000 ms (medium sessions)
  - Packet counts: 1,000-10,000 forward, 50-500 backward
  - Throughput: 1M-10M bytes/s (moderate)
  - Inter-arrival time: 2-10 ms (regular intervals)
  - **Samples:** 50

### 5. **Brute Force**
- **Purpose:** Credential cracking attempts
- **Characteristics:**
  - Port: 22 (SSH)
  - Duration: 500-5,000 ms (short attempts)
  - Packet counts: 10-100 forward, 10-100 backward (bidirectional)
  - Throughput: 1,000-10,000 bytes/s (low)
  - Inter-arrival time: 5-50 ms (regular intervals)
  - **Samples:** 50

### 6. **Reconnaissance (Recon)**
- **Purpose:** Network scanning and information gathering
- **Characteristics:**
  - Ports: 22, 80, 443, 3389 (various ports scanned)
  - Duration: 50-500 ms (very short)
  - Packet counts: 1 forward, 0 backward (port closed)
  - Throughput: 50-200 bytes/s (very low)
  - Inter-arrival time: 50-500 ms (longer intervals)
  - **Samples:** 50

### 7. **Spoofing**
- **Purpose:** IP/identity spoofing attacks
- **Characteristics:**
  - Port: 53 (DNS)
  - Duration: 1,000-10,000 ms (medium)
  - Packet counts: 100-1,000 forward, 0 backward
  - Throughput: 10,000-100,000 bytes/s (moderate)
  - Inter-arrival time: 2-20 ms (regular but suspicious)
  - **Samples:** 50

### 8. **IIoT Attack (CICAPT-IIOT Specific)**
- **Purpose:** Industrial IoT system attacks (Modbus protocol)
- **Characteristics:**
  - Port: 502 (Modbus - industrial protocol)
  - Duration: 5,000-50,000 ms (longer sessions)
  - Packet counts: 1,000-10,000 forward, 1,000-10,000 backward
  - Throughput: 5M-50M bytes/s (high)
  - Inter-arrival time: 2-20 ms (regular intervals)
  - **Samples:** 50

---

## 📈 **Test Data Summary**

| Threat Type | Samples | Total Features | Label |
|------------|---------|----------------|-------|
| Normal Traffic | 100 | 145 | BENIGN |
| DoS | 50 | 145 | ATTACK |
| DDoS | 50 | 145 | ATTACK |
| Mirai Botnet | 50 | 145 | ATTACK |
| Brute Force | 50 | 145 | ATTACK |
| Recon | 50 | 145 | ATTACK |
| Spoofing | 50 | 145 | ATTACK |
| IIoT Attack | 50 | 145 | ATTACK |
| **TOTAL** | **450** | **145** | **Mixed** |

---

## 🔍 **How to Verify New Datasets Are Being Used**

### Method 1: Check API Logs
```bash
# View real-time API logs
tail -f logs/api.log | grep -i "new\|iot\|cicapt"

# Check startup logs
cat logs/api_startup.log | grep -i "new"
```

### Method 2: Check System Info Endpoint
```bash
curl -H "X-API-Key: dev-key-123" http://localhost:5001/system/info | python3 -m json.tool | grep -A 10 "models_loaded"
```

### Method 3: Check Model Files
```bash
# List new model files
ls -lh models/*new_datasets.pkl

# Check file modification dates (should be recent)
stat models/random_forest_new_datasets.pkl
```

### Method 4: Check Training Results
```bash
# View training results
cat results/training_results_new_datasets_*.json | python3 -m json.tool
```

### Method 5: Check Feature Count
```bash
python3 << EOF
import joblib
old = joblib.load('models/feature_names.pkl')
new = joblib.load('models/feature_names_new_datasets.pkl')
print(f"Old CICIDS2017: {len(old)} features")
print(f"New IoT-IDAD + CICAPT: {len(new)} features")
print(f"Difference: +{len(new) - len(old)} features")
EOF
```

---

## 📝 **Dashboard Evidence**

When you click "Send Test Data" on the dashboard, you can verify:

1. **Model Performance Monitor** shows:
   - Random Forest: HEALTHY (using new model)
   - XGBoost: HEALTHY (using new model)
   - Isolation Forest: HEALTHY (using new model) ✅ **FIXED**

2. **Threat Detection** shows various threat levels based on the attack types sent

3. **Real-time Threat Scores** chart updates with threat scores from new models

---

## ✅ **Isolation Forest Status**

**Status:** ✅ **FIXED AND WORKING**

**Previous Issue:** The model was saved with a custom wrapper class that couldn't be loaded.

**Solution:** Retrained and saved as a pure sklearn `IsolationForest` model that can be loaded directly.

**Current Status:** 
- ✅ Model loads successfully
- ✅ Predictions work correctly
- ✅ Returns -1 for anomalies (attacks), 1 for normal traffic
- ✅ Integrated with API and dashboard

**Verification:**
```bash
# Check model file
ls -lh models/isolation_forest_new_datasets.pkl

# Test model loading
python3 -c "import joblib; m=joblib.load('models/isolation_forest_new_datasets.pkl'); print('Type:', type(m).__name__); print('Has predict:', hasattr(m, 'predict'))"
```

---

## 📚 **Dataset Sources**

1. **CIC IoT-IDAD Dataset 2024**
   - Source: Canadian Institute for Cybersecurity
   - Focus: IoT device attack detection
   - Attacks: DoS, DDoS, Mirai, Brute Force, Recon, Spoofing

2. **CICAPT-IIOT**
   - Source: Canadian Institute for Cybersecurity
   - Focus: Industrial IoT (IIoT) attack detection
   - Protocol: Modbus (port 502)
   - Attacks: IIoT-specific attacks

3. **CICIDS2017** (Old dataset, used for comparison)
   - Source: Canadian Institute for Cybersecurity
   - Focus: General network intrusion detection
   - Used as baseline for comparison

---

## 🎓 **For Your Presentation**

### Key Points to Highlight:

1. **✅ New Datasets Integrated:**
   - IoT-IDAD 2024 (latest IoT attack dataset)
   - CICAPT-IIOT (Industrial IoT attacks)
   - 145 features vs 78 in old models

2. **✅ Comprehensive Threat Coverage:**
   - 8 different attack types simulated
   - 450 test samples covering various scenarios
   - Both benign and malicious traffic patterns

3. **✅ Production-Ready System:**
   - Real-time dashboard with live updates
   - Multiple ML models (Random Forest, XGBoost, Isolation Forest)
   - Comprehensive logging and monitoring

4. **✅ Evidence of New Models:**
   - Model files: `*_new_datasets.pkl`
   - API logs show "NEW (IoT-IDAD + CICAPT)" labels
   - Feature count: 145 (vs 78 old)

### Demonstration Steps:

1. **Show API logs:**
   ```bash
   tail -20 logs/api_startup.log
   ```

2. **Show model files:**
   ```bash
   ls -lh models/*new_datasets.pkl
   ```

3. **Show test data:**
   ```bash
   python3 -c "import json; data=json.load(open('test_data/test_data_new_models.json')); print(f'Total samples: {len(data)}'); print('Attack types:', set([s['attack_type'] for s in data]))"
   ```

4. **Show dashboard:**
   - Click "Send Test Data"
   - Show threat detection in real-time
   - Show model performance metrics

---

## 📞 **Quick Reference Commands**

```bash
# View logs in real-time
tail -f logs/api.log

# Check which models are loaded
curl -H "X-API-Key: dev-key-123" http://localhost:5001/system/info

# View test data summary
python3 -c "import json; data=json.load(open('test_data/test_data_new_models.json')); print(f'Samples: {len(data)}'); from collections import Counter; print('Attack types:', dict(Counter([s['attack_type'] for s in data])))"

# Check feature count
python3 -c "import joblib; print('New features:', len(joblib.load('models/feature_names_new_datasets.pkl')))"
```

---

**Last Updated:** November 12, 2025
**System Status:** ✅ New models active, **3/3 models healthy** (All models working correctly)

