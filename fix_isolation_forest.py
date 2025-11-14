#!/usr/bin/env python3
"""
Fix Isolation Forest Model - Retrain and save as sklearn model directly
"""

import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader_multi import MultiDatasetLoader
from utils.feature_aligner import align_multiple_datasets
from preprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split

def main():
    print("=" * 80)
    print("FIXING ISOLATION FOREST MODEL")
    print("=" * 80)
    
    # Load the same data that was used for training
    print("\n1. Loading datasets...")
    loader = MultiDatasetLoader()
    
    datasets = []
    for key in ["cic_iot_2024", "cicapt_iiot"]:
        try:
            print(f"  Loading {key}...")
            df = loader.load_dataset(key, sample_size=50000)
            datasets.append(df)
            print(f"    Loaded: {len(df)} samples")
        except Exception as e:
            print(f"    Error loading {key}: {e}")
            continue
    
    if not datasets:
        print("❌ No datasets loaded")
        return
    
    # Align features
    print("\n2. Aligning features...")
    combined_df, common_features = align_multiple_datasets(datasets, fill_missing="zero")
    print(f"  Combined: {len(combined_df)} samples, {len(common_features)} features")
    
    # Preprocess
    print("\n3. Preprocessing...")
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.clean_data(combined_df)
    
    label_col = " Label" if " Label" in df_cleaned.columns else "Label"
    X = df_cleaned[common_features].copy()
    y = (df_cleaned[label_col] != "BENIGN").astype(int)
    
    # Handle missing values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load the scaler that was used
    print("\n4. Loading scaler...")
    scaler_path = Path("models/scaler_new_datasets.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print("  ✅ Loaded existing scaler")
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, scaler_path)
        print("  ✅ Created and saved new scaler")
    
    # Scale data
    X_train_scaled = scaler.transform(X_train) if not isinstance(X_train, np.ndarray) or X_train.dtype != np.float64 else X_train
    if not hasattr(X_train, 'shape') or X_train.shape[1] != len(common_features):
        X_train_scaled = scaler.transform(X_train.values) if hasattr(X_train, 'values') else scaler.transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    # Train Isolation Forest on normal data only
    print("\n5. Training Isolation Forest...")
    X_normal = X_train_scaled[y_train == 0]
    print(f"  Normal samples for training: {len(X_normal)}")
    
    # Create and train sklearn IsolationForest directly
    isolation_forest = IsolationForest(
        contamination=0.1,
        n_estimators=100,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training model...")
    isolation_forest.fit(X_normal)
    print("  ✅ Model trained")
    
    # Test the model
    print("\n6. Testing model...")
    predictions = isolation_forest.predict(X_test)
    # Convert -1 (anomaly) to 1 (attack), 1 (normal) to 0 (benign)
    binary_predictions = (predictions == -1).astype(int)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, binary_predictions)
    print(f"  Test Accuracy: {accuracy:.4f}")
    
    # Save the model directly (not wrapped)
    print("\n7. Saving model...")
    model_path = Path("models/isolation_forest_new_datasets.pkl")
    joblib.dump(isolation_forest, model_path)
    print(f"  ✅ Model saved to {model_path}")
    
    # Verify it can be loaded
    print("\n8. Verifying model can be loaded...")
    try:
        loaded_model = joblib.load(model_path)
        print(f"  ✅ Model loaded successfully")
        print(f"  Type: {type(loaded_model).__name__}")
        print(f"  Has predict: {hasattr(loaded_model, 'predict')}")
        
        # Test prediction
        test_pred = loaded_model.predict(X_test[:1])
        print(f"  ✅ Test prediction works: {test_pred}")
        
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return
    
    print("\n" + "=" * 80)
    print("✅ ISOLATION FOREST MODEL FIXED!")
    print("=" * 80)
    print("\nThe model is now saved as a pure sklearn IsolationForest model")
    print("and can be loaded without custom class dependencies.")
    print("\nNext: Restart the API to load the fixed model.")

if __name__ == "__main__":
    main()

