# AI Cybersecurity Tool

End-to-end ML pipeline for network threat detection using CICIDS2017 with real-time dashboard and advanced explainability features.

## Features
- **Week 1**: Baseline pipeline (RandomForest), preprocessing, artifacts saved to `data/processed/` and `models/`
- **Week 2**: Supervised (XGBoost), anomaly detection (IsolationForest), ensemble combiner, comprehensive evaluation, performance monitoring
- **Week 3**: Flask API + SocketIO with real-time alerts, batch processing endpoint
- **Week 4**: Complete React dashboard with real-time threat monitoring, interactive charts, and live system health tracking
- **Advanced Features**:
  - Feature importance analysis and attention visualization
  - Model performance monitoring with real-time metrics
  - SSL enhancement with SimCLR-style contrastive learning
  - Modern glassmorphism UI with animated effects

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

## Advanced Model Components

### SSL Enhancement
- `src/models/ssl_enhancement.py`: SimCLR-style self-supervised learning
- `src/models/ssl_encoder.py`: SSL encoder implementation
- `src/models/integrate_ssl.py`: SSL integration with existing models
- `src/models/ssl_api_integration.py`: Production-ready SSL API functions
- `train_ssl_enhanced.py`: SSL training pipeline
- `compare_ssl_performance.py`: Performance comparison tools
- `test_ssl_module.py`: SSL module validation tests

### Explainability & Monitoring
- `src/models/attention_explainer.py`: Attention-based feature importance explanation
- `api/endpoints/explain.py`: Explanation API endpoint module
- Model performance tracking with running averages
- Feature importance visualization

## API (Week 3)
Start server:
```bash
source venv/bin/activate
python api/app.py
```
**Server runs on port 5001** (avoiding macOS Control Center conflict on 5000)

### Core Endpoints:
- `GET /health` – Health check
- `POST /predict` – Single prediction with feature analysis
- `POST /batch/predict` – Batch predictions
- `GET /stats` – Threat statistics and history
- `GET /alerts` – Recent security alerts
- `GET /system/info` – System performance metrics

### Advanced Endpoints:
- `POST /explain` – Get feature importance explanation for predictions
- `GET /models/performance` – Real-time model performance metrics
  - Predictions count per model
  - Average confidence scores
  - Response times
  - Model health status

### Authentication:
All endpoints require `X-API-Key` header:
```bash
curl -H "X-API-Key: dev-key-123" http://localhost:5001/health
```

Requirements:
- Expects models/artifacts in `models/`:
  - RandomForest: `baseline_model.pkl` (or `baseline_rf_model.pkl`)
  - Optional: `xgboost_model.pkl`, `isolation_forest.pkl`
  - Preprocessors: `scaler.pkl`, `feature_names.pkl` (or `selected_features.pkl`)

WebSocket:
- Socket.IO event `new_alert` emitted for High/Critical detections.

## Dashboard (Week 4)
Start React dashboard:
```bash
cd frontend/cybersecurity-dashboard
npm install
npm start
```
**Dashboard runs on port 3000**

### Core Features:
- **Real-time threat monitoring** with live updates via WebSocket
- **Interactive charts** for threat scores and distribution
- **System health monitoring** with CPU/memory usage
- **Alert system** with threat level classification (None, Low, Medium, High, Critical)
- **Material-UI components** for modern, responsive interface

### Advanced Features:
- **Feature Attention Visualizer** - Shows which features contributed most to predictions with:
  - Bar chart visualization of feature importance
  - Color-coded importance levels
  - Natural language explanation of predictions
  - Top 3 most influential features highlighted

- **Model Performance Monitor** - Real-time model metrics including:
  - Live predictions count per model
  - Average confidence scores
  - Response time tracking
  - Model health status indicators
  - Contribution weight visualization

- **Modern UI/UX Enhancements**:
  - Glassmorphism effects with backdrop blur
  - Animated gradient borders and glow effects
  - Shimmer animations on metric cards
  - Pulsing alerts for critical threats
  - Multi-layered background gradients
  - Smooth transitions and hover effects

### Testing the System
1. Start both servers (API on 5001, Dashboard on 3000)
2. Visit `http://localhost:3000` for the dashboard
3. Test API with realistic network features:
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Destination Port": 80,
      "Flow Duration": 1000000,
      "Total Fwd Packets": 10000,
      "Total Backward Packets": 10000,
      "source_ip": "192.168.1.100",
      "attack_type": "DDoS_Test"
    }
  }'
```

## SSL Enhancement Usage

### Quick SSL Setup
```bash
# Install SSL dependencies
pip install torch torchvision tqdm

# Test SSL module
python test_ssl_module.py

# Train SSL encoder (optional enhancement)
python train_ssl_enhanced.py --mode ssl-only --epochs 50

# Compare performance (quick test)
python quick_ssl_performance_test.py

# Compare performance (comprehensive - may take longer)
python compare_ssl_performance.py
```

### SSL Features
- **Self-supervised learning** with SimCLR-style contrastive learning
- **Network-specific augmentation** (noise, dropout, scaling, temporal shifts)
- **Non-destructive integration** - preserves existing 99.97% accuracy baseline
- **Production-ready** - API-compatible with graceful fallback
- **Performance monitoring** - comprehensive comparison tools

See `SSL_ENHANCEMENT_README.md` for detailed documentation.

## Testing

### Test Scripts
- `test_simple_predictions.py`: Basic prediction testing
- `test_attention_api.py`: Attention/explanation endpoint testing
- `test_model_performance_api.py`: Model performance monitoring tests
- `test_real_cicids_predictions.py`: Real CICIDS2017 data predictions
- `test_attention_with_real_features.py`: Feature importance with real data
- `tests/test_attention_explainer.py`: Unit tests for attention explainer

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
python test_simple_predictions.py

# Test API endpoints
python test_attention_api.py
python test_model_performance_api.py
```

## Documentation
- Main README: `README.md` (this file)
- SSL Enhancement: `SSL_ENHANCEMENT_README.md`
- Test Results: `TEST_RESULTS_REPORT.md`
- Verification: `VERIFICATION_REPORT.md`
- Progress log: `docs/PROGRESS.md`
- Changelog: `CHANGELOG.md`

## Project Structure
```
ai-cybersecurity-tool/
├── api/                          # Flask API
│   ├── app.py                   # Main API server
│   └── endpoints/               # Modular endpoints
├── frontend/                     # React dashboard
│   └── cybersecurity-dashboard/
│       ├── src/
│       │   ├── components/      # React components
│       │   ├── theme.js         # MUI dark theme
│       │   └── App.js
├── src/
│   ├── models/                  # ML models
│   │   ├── attention_explainer.py
│   │   ├── ssl_encoder.py
│   │   ├── ensemble_model.py
│   │   └── ...
│   └── evaluation/              # Evaluation tools
├── data/                        # Data storage
│   ├── cicids2017/             # CICIDS2017 dataset
│   └── processed/              # Processed data
├── models/                      # Saved models
└── tests/                       # Test suite
```
