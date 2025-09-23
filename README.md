# AI Cybersecurity Tool

End-to-end ML pipeline for network threat detection using CICIDS2017.

## Features
- Week 1: Baseline pipeline (RandomForest), preprocessing, artifacts saved to `data/processed/` and `models/`
- Week 2: Supervised (XGBoost), anomaly detection (IsolationForest), ensemble combiner, comprehensive evaluation, performance monitoring
- Week 3: Flask API + SocketIO with real-time alerts, batch processing endpoint

## Setup
```bash
# From project root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
- Place CICIDS2017 CSVs under `data/cicids2017/MachineLearningCCSV/`.
- Example used in code: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`.

## Run Baseline (Week 1)
```bash
source venv/bin/activate
python run_week1.py
```
Artifacts:
- Processed arrays: `data/processed/{X_train,X_test,y_train,y_test}.npy`
- Model: `models/baseline_model.pkl`
- Preprocessors: `models/{scaler.pkl,feature_names.pkl}`

## Models (Week 2)
- `src/models/xgboost_model.py`: `XGBoostDetector`
- `src/models/anomaly_detector.py`: `AnomalyDetector` (IsolationForest)
- `src/models/ensemble_model.py`: `EnsembleDetector`
- `src/evaluation/model_evaluator.py`: metrics, confusion matrix, model comparison
- `src/evaluation/performance_monitor.py`: real-time performance metrics

## API (Week 3)
Start server:
```bash
source venv/bin/activate
python api/app.py
```
Endpoints:
- `GET /health` – health check
- `POST /predict` – single prediction; JSON { features: { ... } }
- `POST /batch/predict` – batch predictions; JSON { logs: [ { ... }, ... ] }
- `GET /stats`, `GET /alerts`, `GET /system/info`

Requirements:
- Expects models/artifacts in `models/`:
  - RandomForest: `baseline_model.pkl` (or `baseline_rf_model.pkl`)
  - Optional: `xgboost_model.pkl`, `isolation_forest.pkl`
  - Preprocessors: `scaler.pkl`, `feature_names.pkl` (or `selected_features.pkl`)

WebSocket:
- Socket.IO event `new_alert` emitted for High/Critical detections.

## Development
- Tests: `pytest`
- Progress log: `docs/PROGRESS.md`
- Changelog: `CHANGELOG.md`
