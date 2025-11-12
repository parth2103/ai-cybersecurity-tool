#!/usr/bin/env python3
"""
Training Script for New Datasets

Trains models on new cybersecurity datasets (CIC IoT-IDAD 2024, CICAPT-IIOT, 
Global Cybersecurity Threats) and compares performance with existing CICIDS2017 models.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader_multi import MultiDatasetLoader
from utils.feature_aligner import FeatureAligner, align_multiple_datasets
from preprocessor import DataPreprocessor
from models.xgboost_model import XGBoostDetector
from models.anomaly_detector import AnomalyDetector
from models.ensemble_model import EnsembleDetector
from evaluation.model_evaluator import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_align_datasets(
    dataset_keys: list,
    sample_size_per_dataset: int = None,
    test_size: float = 0.2
):
    """
    Load and align multiple datasets
    
    Args:
        dataset_keys: List of dataset keys to load
        sample_size_per_dataset: Maximum samples per dataset
        test_size: Test set size fraction
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    loader = MultiDatasetLoader()
    
    # Load datasets
    print(f"\n{'=' * 80}")
    print("LOADING DATASETS")
    print(f"{'=' * 80}")
    
    datasets = []
    for key in dataset_keys:
        try:
            print(f"\nLoading {key}...")
            df = loader.load_dataset(key, sample_size=sample_size_per_dataset)
            datasets.append(df)
            print(f"  Loaded: {len(df)} samples")
        except Exception as e:
            logger.error(f"Error loading {key}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No datasets loaded successfully")
    
    # Align features
    print(f"\n{'=' * 80}")
    print("ALIGNING FEATURES")
    print(f"{'=' * 80}")
    
    combined_df, common_features = align_multiple_datasets(datasets, fill_missing="zero")
    print(f"Combined dataset: {len(combined_df)} samples")
    print(f"Common features: {len(common_features)}")
    
    # Prepare features and labels
    print(f"\n{'=' * 80}")
    print("PREPROCESSING DATA")
    print(f"{'=' * 80}")
    
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.clean_data(combined_df)
    
    # Get label column
    label_col = " Label" if " Label" in df_cleaned.columns else "Label"
    if label_col not in df_cleaned.columns:
        raise ValueError("Label column not found in combined dataset")
    
    # Prepare features (use aligned features)
    X = df_cleaned[common_features].copy()
    y = (df_cleaned[label_col] != "BENIGN").astype(int)
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Attack ratio: {y.mean():.2%}")
    
    # Handle missing values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler and feature names
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(scaler, models_dir / "scaler_new_datasets.pkl")
    joblib.dump(common_features, models_dir / "feature_names_new_datasets.pkl")
    
    print(f"\n✅ Data prepared: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test, common_features, scaler


def train_models(X_train, X_test, y_train, y_test):
    """
    Train models on new datasets
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        
    Returns:
        Dictionary of trained models and their performance
    """
    print(f"\n{'=' * 80}")
    print("TRAINING MODELS")
    print(f"{'=' * 80}")
    
    results = {}
    models = {}
    
    # Train XGBoost
    print("\n1. Training XGBoost...")
    try:
        xgb_model = XGBoostDetector()
        xgb_model.train(X_train, y_train, X_test, y_test)
        
        y_pred = (xgb_model.predict_proba(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        
        models['xgboost'] = xgb_model
        results['xgboost'] = {
            'accuracy': float(accuracy),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Save model
        joblib.dump(xgb_model, Path("models/xgboost_model_new_datasets.pkl"))
        
        print(f"  ✅ XGBoost Accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error training XGBoost: {e}", exc_info=True)
    
    # Train Isolation Forest (Anomaly Detector)
    print("\n2. Training Isolation Forest...")
    try:
        # Use normal data only for training
        X_normal = X_train[y_train == 0]
        
        if len(X_normal) > 0:
            anomaly_detector = AnomalyDetector(contamination=0.1)
            anomaly_detector.train(X_normal)
            
            y_pred, scores = anomaly_detector.detect_anomalies(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            models['isolation_forest'] = anomaly_detector
            results['isolation_forest'] = {
                'accuracy': float(accuracy),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Save model
            joblib.dump(anomaly_detector, Path("models/isolation_forest_new_datasets.pkl"))
            
            print(f"  ✅ Isolation Forest Accuracy: {accuracy:.4f}")
        else:
            print("  ⚠️  No normal samples found for training")
    except Exception as e:
        logger.error(f"Error training Isolation Forest: {e}", exc_info=True)
    
    # Train Random Forest (baseline)
    print("\n3. Training Random Forest...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        models['random_forest'] = rf_model
        results['random_forest'] = {
            'accuracy': float(accuracy),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Save model
        joblib.dump(rf_model, Path("models/random_forest_new_datasets.pkl"))
        
        print(f"  ✅ Random Forest Accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error training Random Forest: {e}", exc_info=True)
    
    # Create Ensemble
    print("\n4. Creating Ensemble Model...")
    try:
        if len(models) > 1:
            ensemble = EnsembleDetector(models)
            ensemble.set_weights({
                'xgboost': 0.5,
                'random_forest': 0.3,
                'isolation_forest': 0.2
            } if 'isolation_forest' in models else {
                'xgboost': 0.6,
                'random_forest': 0.4
            })
            
            # Evaluate ensemble
            y_pred_proba = ensemble.predict_ensemble(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['ensemble'] = {
                'accuracy': float(accuracy),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Save ensemble
            joblib.dump(ensemble, Path("models/ensemble_model_new_datasets.pkl"))
            
            print(f"  ✅ Ensemble Accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error creating ensemble: {e}", exc_info=True)
    
    return models, results


def compare_with_old_models(X_test, y_test, feature_names_new):
    """
    Compare new models with old CICIDS2017 models
    
    Args:
        X_test: Test data from new datasets
        y_test: Test labels
        feature_names_new: Feature names from new datasets
    """
    print(f"\n{'=' * 80}")
    print("COMPARING WITH OLD MODELS")
    print(f"{'=' * 80}")
    
    # Load old models and feature names
    models_dir = Path("models")
    
    try:
        # Load old feature names
        feature_names_old = joblib.load(models_dir / "feature_names.pkl")
        scaler_old = joblib.load(models_dir / "scaler.pkl")
        
        # Load old models
        rf_old = joblib.load(models_dir / "baseline_model.pkl")
        xgb_old = joblib.load(models_dir / "xgboost_model.pkl")
        
        print(f"Old features: {len(feature_names_old)}")
        print(f"New features: {len(feature_names_new)}")
        
        # Align test data to old feature set
        X_test_df = pd.DataFrame(X_test, columns=feature_names_new)
        
        # Find common features
        common_features = set(feature_names_old) & set(feature_names_new)
        print(f"Common features: {len(common_features)}")
        
        if len(common_features) < 10:
            print("⚠️  Too few common features, cannot compare reliably")
            return
        
        # Align to old features
        X_test_aligned = np.zeros((len(X_test), len(feature_names_old)))
        for i, feat in enumerate(feature_names_old):
            if feat in feature_names_new:
                feat_idx = feature_names_new.index(feat)
                X_test_aligned[:, i] = X_test[:, feat_idx]
        
        # Scale with old scaler
        X_test_aligned_scaled = scaler_old.transform(X_test_aligned)
        
        # Evaluate old models
        print("\nEvaluating old models on new data:")
        
        # Random Forest
        y_pred_rf = rf_old.predict(X_test_aligned_scaled)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"  Old Random Forest: {acc_rf:.4f}")
        
        # XGBoost
        y_pred_xgb = (xgb_old.predict_proba(X_test_aligned_scaled)[:, 1] > 0.5).astype(int)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        print(f"  Old XGBoost: {acc_xgb:.4f}")
        
    except Exception as e:
        logger.error(f"Error comparing with old models: {e}", exc_info=True)
        print("⚠️  Could not compare with old models (they may not exist)")


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("TRAINING ON NEW DATASETS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Configuration
    # Note: global_threats is excluded - it has no labels and only 4 numeric features
    # It's metadata/statistics, not suitable for network traffic ML training
    DATASETS = [
        "cic_iot_2024",
        "cicapt_iiot",
        # "global_threats",  # Excluded - no labels, not suitable for ML training
    ]
    
    # Optional: Include CICIDS2017 for comparison with new datasets
    # DATASETS_WITH_CICIDS = ["cicids2017"] + DATASETS
    
    SAMPLE_SIZE_PER_DATASET = 50000  # Adjust based on available memory
    TEST_SIZE = 0.2
    
    try:
        # Load and align datasets
        X_train, X_test, y_train, y_test, feature_names, scaler = load_and_align_datasets(
            dataset_keys=DATASETS,
            sample_size_per_dataset=SAMPLE_SIZE_PER_DATASET,
            test_size=TEST_SIZE
        )
        
        # Train models
        models, results = train_models(X_train, X_test, y_train, y_test)
        
        # Compare with old models
        compare_with_old_models(X_test, y_test, feature_names)
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"training_results_new_datasets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to Python types for JSON
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'accuracy': float(result['accuracy']),
                'classification_report': {
                    k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                        for kk, vv in v.items()}
                    if isinstance(v, dict) else v
                    for k, v in result['classification_report'].items()
                }
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n{'=' * 80}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 80}")
        print(f"Results saved to: {results_file}")
        print("\nModel Performance Summary:")
        for model_name, result in results.items():
            print(f"  {model_name}: {result['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

