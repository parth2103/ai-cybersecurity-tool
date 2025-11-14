#!/usr/bin/env python3
"""
Accuracy verification for new models
Tests accuracy on test data and compares with training results
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader_multi import MultiDatasetLoader
from src.utils.feature_aligner import FeatureAligner

MODELS_DIR = Path(__file__).parent / "models"


def load_models():
    """Load new models"""
    models = {}
    
    # Random Forest
    rf_path = MODELS_DIR / "random_forest_new_datasets.pkl"
    if rf_path.exists():
        models['random_forest'] = joblib.load(rf_path)
    
    # XGBoost
    xgb_path = MODELS_DIR / "xgboost_model_new_datasets.pkl"
    if xgb_path.exists():
        models['xgboost'] = joblib.load(xgb_path)
    
    # Isolation Forest
    if_path = MODELS_DIR / "isolation_forest_new_datasets.pkl"
    if if_path.exists():
        models['isolation_forest'] = joblib.load(if_path)
    
    return models


def load_test_data(sample_size=5000):
    """Load test data from new datasets"""
    loader = MultiDatasetLoader()
    aligner = FeatureAligner()
    
    # Load datasets
    datasets = []
    for key in ["cic_iot_2024", "cicapt_iiot"]:
        try:
            df = loader.load_dataset(key, sample_size=sample_size, file_limit=2)
            if df is not None and len(df) > 0:
                datasets.append(df)
        except:
            continue
    
    if not datasets:
        return None, None, None
    
    # Align features
    common_features = aligner.find_common_features(datasets)
    if not common_features:
        return None, None, None
    
    # Combine and prepare
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Get label column
    label_col = " Label" if " Label" in combined_df.columns else "Label"
    if label_col not in combined_df.columns:
        return None, None, None
    
    # Prepare features
    X = combined_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = (combined_df[label_col] != "BENIGN").astype(int)
    
    # Load scaler
    scaler_path = MODELS_DIR / "scaler_new_datasets.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    return X_scaled, y.values, common_features


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model"""
    try:
        # Get predictions
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_pred_proba = proba[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            elif proba.ndim == 1:
                y_pred_proba = proba
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = None
        elif hasattr(model, 'detect_anomalies'):
            y_pred, scores = model.detect_anomalies(X_test)
            y_pred_proba = None
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def test_accuracy():
    """Test model accuracy"""
    print("=" * 60)
    print("MODEL ACCURACY VERIFICATION")
    print("=" * 60)
    
    # Load models
    print("\n1. Loading models...")
    models = load_models()
    if not models:
        print("❌ No models found")
        return
    
    print(f"   Loaded {len(models)} models: {', '.join(models.keys())}")
    
    # Load test data
    print("\n2. Loading test data...")
    X_test, y_test, features = load_test_data(sample_size=5000)
    if X_test is None:
        print("❌ Could not load test data")
        return
    
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(features)}")
    print(f"   Benign: {(y_test == 0).sum()}, Attack: {(y_test == 1).sum()}")
    
    # Expected accuracies from training
    expected_accuracies = {
        "random_forest": 1.0,  # 100%
        "xgboost": 1.0,  # 100%
        "isolation_forest": 0.9497  # 94.97%
    }
    
    # Evaluate each model
    print("\n3. Evaluating models...")
    print("-" * 60)
    
    results = {}
    for model_name, model in models.items():
        print(f"\n{model_name.upper()}:")
        result = evaluate_model(model, X_test, y_test, model_name)
        
        if result["success"]:
            accuracy = result["accuracy"]
            precision = result["precision"]
            recall = result["recall"]
            f1 = result["f1_score"]
            cm = result["confusion_matrix"]
            
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    [[{cm[0][0]}, {cm[0][1]}],")
            print(f"     [{cm[1][0]}, {cm[1][1]}]]")
            
            # Compare with expected
            expected = expected_accuracies.get(model_name, 0.95)
            if accuracy >= expected * 0.95:  # Allow 5% tolerance
                print(f"  ✅ PASS: Accuracy ({accuracy:.4f}) meets expectation (~{expected:.4f})")
            else:
                print(f"  ⚠️  WARNING: Accuracy ({accuracy:.4f}) below expectation ({expected:.4f})")
            
            results[model_name] = result
        else:
            print(f"  ❌ Error: {result.get('error', 'Unknown')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ACCURACY TEST SUMMARY")
    print("=" * 60)
    
    for model_name, result in results.items():
        if result["success"]:
            accuracy = result["accuracy"]
            print(f"{model_name.upper()}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n✅ Accuracy verification complete!")


if __name__ == "__main__":
    test_accuracy()

