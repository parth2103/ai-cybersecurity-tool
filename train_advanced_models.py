#!/usr/bin/env python3
"""
Advanced Model Training Script for AI Cybersecurity Tool
Trains XGBoost, Isolation Forest, and Ensemble models
"""

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')

from models.xgboost_model import XGBoostDetector
from models.anomaly_detector import AnomalyDetector
from models.ensemble_model import EnsembleDetector
from evaluation.model_evaluator import ModelEvaluator
from evaluation.performance_monitor import PerformanceMonitor

def load_processed_data():
    """Load processed training data"""
    project_root = Path(__file__).parent
    processed_dir = project_root / 'data' / 'processed'
    
    print("📊 Loading processed data...")
    X_train = np.load(processed_dir / 'X_train.npy')
    X_test = np.load(processed_dir / 'X_test.npy')
    y_train = np.load(processed_dir / 'y_train.npy')
    y_test = np.load(processed_dir / 'y_test.npy')
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Label distribution - Train: {np.unique(y_train, return_counts=True)}")
    print(f"Label distribution - Test: {np.unique(y_test, return_counts=True)}")
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    print("\n🚀 Training XGBoost Model...")
    print("=" * 50)
    
    # Initialize XGBoost detector
    xgb_detector = XGBoostDetector()
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model
    start_time = time.time()
    model = xgb_detector.train(X_train_split, y_train_split, X_val_split, y_val_split)
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f}s")
    
    # Optimize threshold
    optimal_threshold = xgb_detector.optimize_threshold(X_val_split, y_val_split)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Evaluate on test set
    y_pred_proba = xgb_detector.predict_proba(X_test)
    y_pred = (y_pred_proba > optimal_threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    # Save model
    models_dir = Path('models')
    joblib.dump(xgb_detector, models_dir / 'xgboost_model.pkl')
    print(f"✅ XGBoost model saved to {models_dir / 'xgboost_model.pkl'}")
    
    return xgb_detector, accuracy

def train_anomaly_detector(X_train, y_train, X_test, y_test):
    """Train Isolation Forest anomaly detector"""
    print("\n🔍 Training Isolation Forest Anomaly Detector...")
    print("=" * 50)
    
    # Use only normal samples for training
    normal_mask = y_train == 0
    X_normal = X_train[normal_mask]
    
    print(f"Normal samples for training: {len(X_normal)}")
    print(f"Total training samples: {len(X_train)}")
    
    # Initialize anomaly detector
    anomaly_detector = AnomalyDetector(contamination=0.1)
    
    # Train on normal data only
    start_time = time.time()
    detector = anomaly_detector.train(X_normal)
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f}s")
    
    # Evaluate on test set
    predictions, scores = detector.detect_anomalies(X_test)
    
    # Convert to binary predictions (1 = anomaly, 0 = normal)
    # Note: Isolation Forest returns -1 for anomalies, 1 for normal
    # We need to flip this to match our labeling (1 = attack, 0 = benign)
    binary_predictions = (predictions == -1).astype(int)
    
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, binary_predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, binary_predictions, target_names=['Benign', 'Attack']))
    
    # Save model
    models_dir = Path('models')
    joblib.dump(anomaly_detector, models_dir / 'isolation_forest.pkl')
    print(f"✅ Isolation Forest model saved to {models_dir / 'isolation_forest.pkl'}")
    
    return anomaly_detector, accuracy

def create_ensemble_model():
    """Create and configure ensemble model"""
    print("\n🎯 Creating Ensemble Model...")
    print("=" * 50)
    
    models_dir = Path('models')
    
    # Load all available models
    models = {}
    
    # Load Random Forest
    rf_path = models_dir / 'baseline_model.pkl'
    if rf_path.exists():
        models['rf'] = joblib.load(rf_path)
        print("✅ Loaded Random Forest model")
    else:
        print("⚠️ Random Forest model not found")
    
    # Load XGBoost
    xgb_path = models_dir / 'xgboost_model.pkl'
    if xgb_path.exists():
        models['xgboost'] = joblib.load(xgb_path)
        print("✅ Loaded XGBoost model")
    else:
        print("⚠️ XGBoost model not found")
    
    # Load Isolation Forest
    if_path = models_dir / 'isolation_forest.pkl'
    if if_path.exists():
        models['isolation_forest'] = joblib.load(if_path)
        print("✅ Loaded Isolation Forest model")
    else:
        print("⚠️ Isolation Forest model not found")
    
    if not models:
        print("❌ No models available for ensemble")
        return None
    
    # Create ensemble detector
    ensemble = EnsembleDetector(models)
    
    # Set weights based on model performance (can be tuned)
    weights = {
        'rf': 0.4,
        'xgboost': 0.4,
        'isolation_forest': 0.2
    }
    
    # Only set weights for available models
    available_weights = {k: v for k, v in weights.items() if k in models}
    ensemble.set_weights(available_weights)
    
    print(f"Ensemble weights: {available_weights}")
    
    # Save ensemble model
    joblib.dump(ensemble, models_dir / 'ensemble_model.pkl')
    print(f"✅ Ensemble model saved to {models_dir / 'ensemble_model.pkl'}")
    
    return ensemble

def evaluate_all_models(X_test, y_test):
    """Evaluate all models comprehensively"""
    print("\n📊 Comprehensive Model Evaluation...")
    print("=" * 50)
    
    evaluator = ModelEvaluator()
    models_dir = Path('models')
    
    # Load models
    models = {}
    model_names = ['baseline_model.pkl', 'xgboost_model.pkl', 'isolation_forest.pkl']
    
    for model_name in model_names:
        model_path = models_dir / model_name
        if model_path.exists():
            name = model_name.replace('.pkl', '').replace('_', ' ').title()
            models[name] = joblib.load(model_path)
            print(f"✅ Loaded {name}")
    
    if not models:
        print("❌ No models found for evaluation")
        return
    
    # Evaluate each model
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        try:
            if hasattr(model, 'predict_proba'):
                # For probabilistic models
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                evaluator.evaluate_model(y_test, y_pred, y_pred_proba, name)
            elif hasattr(model, 'detect_anomalies'):
                # For anomaly detection models
                predictions, scores = model.detect_anomalies(X_test)
                y_pred = (predictions == -1).astype(int)
                # Convert scores to probabilities (normalize to 0-1)
                y_pred_proba = (scores - scores.min()) / (scores.max() - scores.min())
                evaluator.evaluate_model(y_test, y_pred, y_pred_proba, name)
            else:
                # For basic models
                y_pred = model.predict(X_test)
                evaluator.evaluate_model(y_test, y_pred, None, name)
                
        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")
    
    # Compare models
    print("\n📈 Model Comparison:")
    evaluator.compare_models()
    
    # Plot comparison (save to file instead of showing)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        evaluator.plot_model_comparison()
        print("📊 Model comparison plots saved to results/")
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")

def main():
    """Main training pipeline"""
    print("🚀 AI Cybersecurity Tool - Advanced Model Training")
    print("=" * 60)
    
    # Check if processed data exists
    processed_dir = Path('data/processed')
    if not processed_dir.exists():
        print("❌ Processed data not found. Please run the preprocessing pipeline first.")
        print("Run: python run_week1.py")
        return
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train XGBoost model
    try:
        xgb_model, xgb_accuracy = train_xgboost_model(X_train, y_train, X_test, y_test)
        print(f"✅ XGBoost training completed - Accuracy: {xgb_accuracy:.4f}")
    except Exception as e:
        print(f"❌ XGBoost training failed: {e}")
        xgb_model = None
    
    # Train Isolation Forest model
    try:
        if_model, if_accuracy = train_anomaly_detector(X_train, y_train, X_test, y_test)
        print(f"✅ Isolation Forest training completed - Accuracy: {if_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Isolation Forest training failed: {e}")
        if_model = None
    
    # Create ensemble model
    try:
        ensemble_model = create_ensemble_model()
        if ensemble_model:
            print("✅ Ensemble model created successfully")
        else:
            print("⚠️ Ensemble model creation failed")
    except Exception as e:
        print(f"❌ Ensemble model creation failed: {e}")
    
    # Comprehensive evaluation
    try:
        evaluate_all_models(X_test, y_test)
        print("✅ Model evaluation completed")
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
    
    # Summary
    print("\n🎉 Advanced Model Training Complete!")
    print("=" * 50)
    print("Models trained:")
    models_dir = Path('models')
    for model_file in models_dir.glob('*.pkl'):
        print(f"  ✅ {model_file.name}")
    
    print("\nNext steps:")
    print("1. Test the models with: python test_comprehensive_threats.py")
    print("2. Start the API with: python api/app.py")
    print("3. View the dashboard at: http://localhost:3000")

if __name__ == "__main__":
    main()
