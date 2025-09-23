# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-09-23
### Added
- Initial scaffold and baseline training pipeline
- Data loader with synthetic fallback
- Preprocessing and model training scripts
- Flask API skeleton with health endpoint
- Docs: PROGRESS.md and PR template

## [0.2.0] - 2025-09-23
### Added
- XGBoostDetector supervised model (src/models/xgboost_model.py)
- IsolationForest-based AnomalyDetector (src/models/anomaly_detector.py)
- EnsembleDetector to combine model outputs (src/models/ensemble_model.py)
- Comprehensive ModelEvaluator utilities (src/evaluation/model_evaluator.py)
- Real-time PerformanceMonitor (src/evaluation/performance_monitor.py)
- Flask API with SocketIO and batch endpoint (api/app.py, api/batch_processor.py)

## [0.3.0] - 2025-09-23
### Added
- Complete React cybersecurity dashboard (frontend/cybersecurity-dashboard/)
- Real-time threat monitoring with WebSocket integration
- Interactive charts for threat scores and distribution
- Live system health monitoring
- Alert system with threat level classification
- Material-UI components for modern interface

### Fixed
- Flask API port conflict (moved from 5000 to 5001)
- React app compilation and module resolution issues
- Circular import issues in Flask API
- Model loading and feature handling in API

### Enhanced
- API endpoints: /predict, /stats, /alerts, /system/info
- Real-time WebSocket communication for live updates
- Comprehensive threat scoring with multiple model support
- Performance monitoring with CPU/memory tracking
- Batch processing capabilities for multiple log analysis

### Tested
- End-to-end pipeline from data loading to dashboard visualization
- AI threat detection with realistic network feature testing
- Real-time dashboard updates and WebSocket communication
- System performance monitoring and health checks
- Threat level classification (None, Low, Medium, High, Critical)
