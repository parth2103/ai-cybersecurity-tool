import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
from pathlib import Path


def train_baseline():
    # Set paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"

    # Load data
    print("Loading processed data...")
    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - start:.2f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))

    # Save model
    joblib.dump(model, models_dir / "baseline_model.pkl")
    print(f"\nModel saved to {models_dir / 'baseline_model.pkl'}")

    return model, accuracy


if __name__ == "__main__":
    train_baseline()
