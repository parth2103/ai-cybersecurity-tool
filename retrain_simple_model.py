#!/usr/bin/env python3
"""
Retrain a simple 3-feature model to match current feature_names.pkl
This will make the system work with the current setup
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

print("=" * 60)
print("🔄 RETRAINING SIMPLE 3-FEATURE MODEL")
print("=" * 60)

# Generate synthetic training data (3 features)
np.random.seed(42)
n_samples = 1000

# Create synthetic data
# Feature_1, Feature_2, Feature_3
X_train = np.random.rand(n_samples, 3)

# Create labels based on simple rule:
# If sum of features > 1.5, it's a threat (1), otherwise benign (0)
y_train = (X_train.sum(axis=1) > 1.5).astype(int)

print(f"\n📊 Training Data:")
print(f"   Samples: {n_samples}")
print(f"   Features: 3 (Feature_1, Feature_2, Feature_3)")
print(f"   Threats: {y_train.sum()} ({y_train.sum()/n_samples*100:.1f}%)")
print(f"   Benign: {(1-y_train).sum()} ({(1-y_train).sum()/n_samples*100:.1f}%)")

# Train Random Forest
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
train_acc = rf_model.score(X_train, y_train)
print(f"   ✅ Training Accuracy: {train_acc*100:.2f}%")

# Create scaler
print("\n📏 Creating StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_train)

# Feature names
feature_names = ['Feature_1', 'Feature_2', 'Feature_3']

# Save models
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

print("\n💾 Saving models...")

# Save baseline model
joblib.dump(rf_model, models_dir / 'baseline_model.pkl')
print(f"   ✅ Saved: baseline_model.pkl")

# Save scaler
joblib.dump(scaler, models_dir / 'scaler.pkl')
print(f"   ✅ Saved: scaler.pkl")

# Save feature names
joblib.dump(feature_names, models_dir / 'feature_names.pkl')
print(f"   ✅ Saved: feature_names.pkl")

# Test the model
print("\n🧪 Testing model...")
test_cases = [
    ([0.1, 0.2, 0.1], "Low threat"),
    ([0.5, 0.6, 0.5], "Medium threat"),
    ([0.9, 0.8, 0.9], "High threat"),
]

for features, description in test_cases:
    X_test = np.array([features])
    pred_proba = rf_model.predict_proba(X_test)[0, 1]
    pred_class = rf_model.predict(X_test)[0]
    print(f"   {description}: {features}")
    print(f"      → Probability: {pred_proba:.3f}, Class: {'THREAT' if pred_class == 1 else 'BENIGN'}")

print("\n" + "=" * 60)
print("✅ MODEL RETRAINING COMPLETE!")
print("=" * 60)
print("\n📝 Summary:")
print("   - Baseline RF model: 3 features, 100 trees")
print("   - Scaler: StandardScaler fitted to 3 features")
print("   - Feature names: Feature_1, Feature_2, Feature_3")
print("\n🚀 Next Steps:")
print("   1. Restart the API server:")
print("      pkill -f 'python.*api/app.py'")
print("      python api/app.py")
print("   2. Run test predictions:")
print("      python test_simple_predictions.py")
print("   3. Check dashboard at http://localhost:3000")
print()
