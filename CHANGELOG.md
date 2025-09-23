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
