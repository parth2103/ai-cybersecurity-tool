#!/usr/bin/env python3
"""
Comparative Analysis: Old CICIDS2017 Models vs New Datasets Models

Compares performance of models trained on:
- Old: CICIDS2017 dataset
- New: CIC IoT-IDAD 2024 + CICAPT-IIOT datasets
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader_multi import MultiDatasetLoader
from utils.feature_aligner import FeatureAligner

# Setup plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_test_data(dataset_keys: List[str], sample_size: int = 10000, min_samples_per_class: int = 1000):
    """Load test data from datasets with balanced classes"""
    print("Loading test data...")
    loader = MultiDatasetLoader()
    
    # Load datasets with larger sample to ensure we get both classes
    datasets = []
    for key in dataset_keys:
        try:
            # Load more data to ensure we get both classes
            df = loader.load_dataset(key, sample_size=sample_size * 2, file_limit=None)
            datasets.append(df)
        except Exception as e:
            print(f"Error loading {key}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No datasets loaded")
    
    # Align features
    aligner = FeatureAligner()
    common_features = aligner.find_common_features(datasets)
    
    # Combine and prepare
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # Get label column
    label_col = " Label" if " Label" in combined_df.columns else "Label"
    if label_col not in combined_df.columns:
        raise ValueError("Label column not found")
    
    # Create binary labels
    y_binary = (combined_df[label_col] != "BENIGN").astype(int)
    combined_df['_binary_label'] = y_binary
    
    # Ensure balanced classes
    benign_samples = combined_df[combined_df['_binary_label'] == 0]
    attack_samples = combined_df[combined_df['_binary_label'] == 1]
    
    # Sample equally from both classes
    min_class_size = min(len(benign_samples), len(attack_samples), sample_size // 2)
    if min_class_size < min_samples_per_class:
        min_class_size = min_samples_per_class
        print(f"Warning: Limited samples per class, using {min_class_size}")
    
    # Sample balanced data
    if len(benign_samples) > 0 and len(attack_samples) > 0:
        benign_sampled = benign_samples.sample(n=min(min_class_size, len(benign_samples)), random_state=42)
        attack_sampled = attack_samples.sample(n=min(min_class_size, len(attack_samples)), random_state=42)
        balanced_df = pd.concat([benign_sampled, attack_sampled], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    else:
        balanced_df = combined_df.sample(n=min(sample_size, len(combined_df)), random_state=42)
    
    print(f"Loaded test data: {len(balanced_df)} samples")
    print(f"  Benign: {(balanced_df['_binary_label'] == 0).sum()}")
    print(f"  Attack: {(balanced_df['_binary_label'] == 1).sum()}")
    
    # Prepare features
    X = balanced_df[common_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = balanced_df['_binary_label'].values
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, common_features, scaler

def evaluate_model(model, X_test, y_test, model_name: str, model_type: str = "sklearn"):
    """Evaluate a model and return metrics"""
    try:
        y_pred_proba = None
        
        if model_type == "xgboost":
            # XGBoost model - check if it's XGBoostDetector or standard XGBoost
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_test)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        y_pred_proba = proba[:, 1]
                        y_pred = (y_pred_proba > 0.5).astype(int)
                    elif proba.ndim == 1:
                        # Single dimension - might be binary probabilities
                        y_pred_proba = proba
                        y_pred = (y_pred_proba > 0.5).astype(int)
                    else:
                        # Fallback
                        y_pred = (proba > 0.5).astype(int).flatten()
                        y_pred_proba = proba.flatten()
                except Exception as e:
                    print(f"Error with XGBoost predict_proba: {e}")
                    # Try model.model.predict_proba if it's a wrapper
                    if hasattr(model, 'model'):
                        try:
                            proba = model.model.predict_proba(X_test)
                            y_pred_proba = proba[:, 1] if proba.ndim == 2 else proba
                            y_pred = (y_pred_proba > 0.5).astype(int)
                        except:
                            return None
                    else:
                        return None
            else:
                return None
                
        elif model_type == "isolation_forest":
            # Isolation Forest (anomaly detector)
            if hasattr(model, 'detect_anomalies'):
                y_pred, scores = model.detect_anomalies(X_test)
                # Convert scores to probabilities (normalize to [0, 1])
                # Lower scores = more anomalous = higher attack probability
                if scores is not None:
                    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                    y_pred_proba = 1 - scores_normalized  # Invert so higher = more attack
            else:
                predictions = model.predict(X_test)
                y_pred = (predictions == -1).astype(int)
                # Try to get decision function for probabilities
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X_test)
                    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                    y_pred_proba = 1 - scores_normalized
        else:
            # Standard sklearn models
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_test)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        y_pred_proba = proba[:, 1]
                except:
                    pass
        
        # Check if we have both classes in predictions and test set
        if len(np.unique(y_test)) < 2:
            print(f"Warning: Test set has only one class for {model_name}")
            # Return None to skip this evaluation
            return None
        
        if len(np.unique(y_pred)) < 2 and len(np.unique(y_test)) == 2:
            print(f"Warning: Model {model_name} predicts only one class")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve (if probabilities available and we have both classes)
        roc_auc = None
        fpr = None
        tpr = None
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
            except:
                pass
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None

def load_old_models():
    """Load old CICIDS2017 models"""
    models_dir = Path("models")
    old_models = {}
    
    try:
        # Load old feature names and scaler
        old_feature_names = joblib.load(models_dir / "feature_names.pkl")
        old_scaler = joblib.load(models_dir / "scaler.pkl")
        
        # Load Random Forest
        if (models_dir / "baseline_model.pkl").exists():
            old_models['random_forest'] = {
                'model': joblib.load(models_dir / "baseline_model.pkl"),
                'type': 'sklearn',
                'feature_names': old_feature_names,
                'scaler': old_scaler
            }
        
        # Load XGBoost
        if (models_dir / "xgboost_model.pkl").exists():
            old_models['xgboost'] = {
                'model': joblib.load(models_dir / "xgboost_model.pkl"),
                'type': 'xgboost',
                'feature_names': old_feature_names,
                'scaler': old_scaler
            }
        
        # Load Isolation Forest
        if (models_dir / "isolation_forest.pkl").exists():
            old_models['isolation_forest'] = {
                'model': joblib.load(models_dir / "isolation_forest.pkl"),
                'type': 'isolation_forest',
                'feature_names': old_feature_names,
                'scaler': old_scaler
            }
        
        print(f"Loaded {len(old_models)} old models")
        return old_models
    except Exception as e:
        print(f"Error loading old models: {e}")
        return {}

def load_new_models():
    """Load new models trained on new datasets"""
    models_dir = Path("models")
    new_models = {}
    
    try:
        # Load new feature names and scaler
        new_feature_names = joblib.load(models_dir / "feature_names_new_datasets.pkl")
        new_scaler = joblib.load(models_dir / "scaler_new_datasets.pkl")
        
        # Load Random Forest
        if (models_dir / "random_forest_new_datasets.pkl").exists():
            new_models['random_forest'] = {
                'model': joblib.load(models_dir / "random_forest_new_datasets.pkl"),
                'type': 'sklearn',
                'feature_names': new_feature_names,
                'scaler': new_scaler
            }
        
        # Load XGBoost
        if (models_dir / "xgboost_model_new_datasets.pkl").exists():
            new_models['xgboost'] = {
                'model': joblib.load(models_dir / "xgboost_model_new_datasets.pkl"),
                'type': 'xgboost',
                'feature_names': new_feature_names,
                'scaler': new_scaler
            }
        
        # Load Isolation Forest
        if (models_dir / "isolation_forest_new_datasets.pkl").exists():
            new_models['isolation_forest'] = {
                'model': joblib.load(models_dir / "isolation_forest_new_datasets.pkl"),
                'type': 'isolation_forest',
                'feature_names': new_feature_names,
                'scaler': new_scaler
            }
        
        print(f"Loaded {len(new_models)} new models")
        return new_models
    except Exception as e:
        print(f"Error loading new models: {e}")
        return {}

def align_features(X, available_features, target_features, scaler, target_scaler):
    """Align features between old and new models"""
    # Create dataframe for alignment
    X_df = pd.DataFrame(X, columns=available_features)
    
    # Align to target features
    X_aligned = np.zeros((len(X), len(target_features)))
    for i, feat in enumerate(target_features):
        if feat in available_features:
            feat_idx = available_features.index(feat)
            X_aligned[:, i] = X[:, feat_idx]
        # else: leave as zero (missing feature)
    
    # Scale with target scaler
    X_scaled = target_scaler.transform(X_aligned)
    
    return X_scaled

def create_comparison_visualizations(results: Dict, output_dir: Path):
    """Create visualization charts for comparison"""
    output_dir.mkdir(exist_ok=True)
    
    # 1. Metrics Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison: Old vs New Datasets', fontsize=16, fontweight='bold')
    
    models = ['random_forest', 'xgboost', 'isolation_forest']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        old_values = []
        new_values = []
        model_names = []
        
        for model in models:
            old_key = f"old_{model}"
            new_key = f"new_{model}"
            
            if old_key in results and new_key in results:
                old_values.append(results[old_key].get(metric, 0))
                new_values.append(results[new_key].get(metric, 0))
                model_names.append(model.replace('_', ' ').title())
        
        if old_values and new_values:
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, old_values, width, label='Old (CICIDS2017)', alpha=0.8)
            bars2 = ax.bar(x + width/2, new_values, width, label='New (IoT-IDAD + CICAPT)', alpha=0.8)
            
            ax.set_ylabel(metric_label, fontsize=12)
            ax.set_title(f'{metric_label} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'metrics_comparison.png'}")
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices: Old vs New Models', fontsize=16, fontweight='bold')
    
    for idx, model in enumerate(models):
        old_key = f"old_{model}"
        new_key = f"new_{model}"
        
        # Old model
        if old_key in results and results[old_key].get('confusion_matrix') is not None:
            cm_old = results[old_key]['confusion_matrix']
            ax = axes[0, idx]
            sns.heatmap(cm_old, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
            ax.set_title(f'Old {model.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # New model
        if new_key in results and results[new_key].get('confusion_matrix') is not None:
            cm_new = results[new_key]['confusion_matrix']
            ax = axes[1, idx]
            sns.heatmap(cm_new, annot=True, fmt='d', cmap='Greens', ax=ax,
                       xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
            ax.set_title(f'New {model.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'confusion_matrices.png'}")
    
    # 3. ROC Curves (if available)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in models:
        old_key = f"old_{model}"
        new_key = f"new_{model}"
        
        if old_key in results and results[old_key].get('roc_auc') is not None:
            fpr = results[old_key]['fpr']
            tpr = results[old_key]['tpr']
            roc_auc = results[old_key]['roc_auc']
            ax.plot(fpr, tpr, label=f'Old {model.replace("_", " ").title()} (AUC = {roc_auc:.3f})',
                   linestyle='--', linewidth=2)
        
        if new_key in results and results[new_key].get('roc_auc') is not None:
            fpr = results[new_key]['fpr']
            tpr = results[new_key]['tpr']
            roc_auc = results[new_key]['roc_auc']
            ax.plot(fpr, tpr, label=f'New {model.replace("_", " ").title()} (AUC = {roc_auc:.3f})',
                   linestyle='-', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'roc_curves.png'}")
    
    # 4. Performance Summary Table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for model in models:
        old_key = f"old_{model}"
        new_key = f"new_{model}"
        
        if old_key in results:
            r = results[old_key]
            table_data.append([
                model.replace('_', ' ').title(),
                'CICIDS2017',
                f"{r.get('accuracy', 0):.4f}",
                f"{r.get('precision', 0):.4f}",
                f"{r.get('recall', 0):.4f}",
                f"{r.get('f1_score', 0):.4f}",
                f"{r.get('roc_auc', 0):.4f}" if r.get('roc_auc') else 'N/A'
            ])
        
        if new_key in results:
            r = results[new_key]
            table_data.append([
                model.replace('_', ' ').title(),
                'IoT-IDAD + CICAPT',
                f"{r.get('accuracy', 0):.4f}",
                f"{r.get('precision', 0):.4f}",
                f"{r.get('recall', 0):.4f}",
                f"{r.get('f1_score', 0):.4f}",
                f"{r.get('roc_auc', 0):.4f}" if r.get('roc_auc') else 'N/A'
            ])
    
    if table_data:
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Performance Metrics Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(output_dir / 'metrics_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir / 'metrics_table.png'}")

def generate_report(results: Dict, output_dir: Path):
    """Generate comprehensive comparison report"""
    report_path = output_dir / f"model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Model Performance Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report compares the performance of models trained on:\n")
        f.write("- **Old Models:** CICIDS2017 dataset\n")
        f.write("- **New Models:** CIC IoT-IDAD 2024 + CICAPT-IIOT datasets\n\n")
        
        f.write("## Models Compared\n\n")
        f.write("1. **Random Forest**\n")
        f.write("2. **XGBoost**\n")
        f.write("3. **Isolation Forest**\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("### Accuracy Comparison\n\n")
        f.write("| Model | Old (CICIDS2017) | New (IoT-IDAD + CICAPT) | Difference |\n")
        f.write("|-------|------------------|--------------------------|------------|\n")
        
        models = ['random_forest', 'xgboost', 'isolation_forest']
        for model in models:
            old_key = f"old_{model}"
            new_key = f"new_{model}"
            
            if old_key in results and new_key in results:
                old_acc = results[old_key].get('accuracy', 0)
                new_acc = results[new_key].get('accuracy', 0)
                diff = new_acc - old_acc
                diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                
                f.write(f"| {model.replace('_', ' ').title()} | {old_acc:.4f} | {new_acc:.4f} | {diff_str} |\n")
        
        f.write("\n### Detailed Metrics\n\n")
        for model in models:
            old_key = f"old_{model}"
            new_key = f"new_{model}"
            
            f.write(f"#### {model.replace('_', ' ').title()}\n\n")
            f.write("| Metric | Old Model | New Model |\n")
            f.write("|--------|-----------|----------|\n")
            
            if old_key in results and new_key in results:
                old_r = results[old_key]
                new_r = results[new_key]
                
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                
                for metric, name in zip(metrics, metric_names):
                    old_val = old_r.get(metric, 0)
                    new_val = new_r.get(metric, 0)
                    f.write(f"| {name} | {old_val:.4f} | {new_val:.4f} |\n")
                
                if old_r.get('roc_auc') and new_r.get('roc_auc'):
                    f.write(f"| ROC-AUC | {old_r['roc_auc']:.4f} | {new_r['roc_auc']:.4f} |\n")
            
            f.write("\n")
        
        f.write("## Visualizations\n\n")
        f.write("1. **Metrics Comparison Chart** - `metrics_comparison.png`\n")
        f.write("2. **Confusion Matrices** - `confusion_matrices.png`\n")
        f.write("3. **ROC Curves** - `roc_curves.png`\n")
        f.write("4. **Metrics Table** - `metrics_table.png`\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Calculate improvements
        improvements = []
        for model in models:
            old_key = f"old_{model}"
            new_key = f"new_{model}"
            
            if old_key in results and new_key in results:
                old_acc = results[old_key].get('accuracy', 0)
                new_acc = results[new_key].get('accuracy', 0)
                improvement = ((new_acc - old_acc) / old_acc * 100) if old_acc > 0 else 0
                improvements.append((model, improvement))
        
        if improvements:
            f.write("### Performance Changes\n\n")
            for model, improvement in improvements:
                if improvement > 0:
                    f.write(f"- **{model.replace('_', ' ').title()}:** Improved by {improvement:.2f}%\n")
                elif improvement < 0:
                    f.write(f"- **{model.replace('_', ' ').title()}:** Decreased by {abs(improvement):.2f}%\n")
                else:
                    f.write(f"- **{model.replace('_', ' ').title()}:** No change\n")
        
        f.write("\n## Conclusions\n\n")
        f.write("### Recommendations\n\n")
        f.write("1. **Model Selection:** Based on the comparison, choose the model with best performance for your use case.\n")
        f.write("2. **Dataset Diversity:** New models trained on multiple datasets may have better generalization.\n")
        f.write("3. **Production Deployment:** Consider deploying the best-performing model to production.\n")
        f.write("4. **Continuous Monitoring:** Monitor model performance in production and retrain as needed.\n\n")
    
    print(f"✅ Saved report: {report_path}")
    return report_path

def main():
    """Main comparison pipeline"""
    print("=" * 80)
    print("MODEL COMPARISON: OLD vs NEW DATASETS")
    print("=" * 80)
    
    # Load models
    print("\n1. Loading models...")
    old_models = load_old_models()
    new_models = load_new_models()
    
    if not old_models or not new_models:
        print("❌ Error: Could not load models")
        return
    
    # Load test data
    print("\n2. Loading test data...")
    
    # Try to use preprocessed test data if available
    processed_dir = Path("data/processed")
    use_preprocessed = False
    
    if (processed_dir / "X_test.npy").exists() and (processed_dir / "y_test.npy").exists():
        print("   Using preprocessed test data from training...")
        try:
            X_test_old = np.load(processed_dir / "X_test.npy")
            y_test_old = np.load(processed_dir / "y_test.npy")
            old_features = joblib.load(Path("models/feature_names.pkl"))
            old_scaler = joblib.load(Path("models/scaler.pkl"))
            
            # Ensure we have both classes
            if len(np.unique(y_test_old)) == 2:
                use_preprocessed = True
                print(f"   Loaded preprocessed data: {len(X_test_old)} samples")
                print(f"   Benign: {(y_test_old == 0).sum()}, Attack: {(y_test_old == 1).sum()}")
        except Exception as e:
            print(f"   Error loading preprocessed data: {e}")
    
    if not use_preprocessed:
        try:
            # Test on CICIDS2017 - need to load files with attacks
            # Load from multiple files to get both benign and attack samples
            loader = MultiDatasetLoader()
            df_cicids = loader.load_dataset("cicids2017", sample_size=20000, file_limit=None)
            
            # Get label column
            label_col = " Label" if " Label" in df_cicids.columns else "Label"
            y_binary = (df_cicids[label_col] != "BENIGN").astype(int)
            
            # Ensure balanced classes
            benign_samples = df_cicids[y_binary == 0].sample(n=min(5000, (y_binary == 0).sum()), random_state=42)
            attack_samples = df_cicids[y_binary == 1].sample(n=min(5000, (y_binary == 1).sum()), random_state=42)
            
            if len(benign_samples) > 0 and len(attack_samples) > 0:
                balanced_df = pd.concat([benign_samples, attack_samples], ignore_index=True)
                balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                balanced_df = df_cicids.sample(n=min(10000, len(df_cicids)), random_state=42)
            
            # Get features
            old_features = joblib.load(Path("models/feature_names.pkl"))
            X_test_old = balanced_df[old_features].fillna(0).replace([np.inf, -np.inf], 0).values
            y_test_old = (balanced_df[label_col] != "BENIGN").astype(int).values
            
            # Scale
            old_scaler = joblib.load(Path("models/scaler.pkl"))
            X_test_old = old_scaler.transform(X_test_old)
            
            print(f"   Loaded CICIDS2017 test data: {len(X_test_old)} samples")
            print(f"   Benign: {(y_test_old == 0).sum()}, Attack: {(y_test_old == 1).sum()}")
        except Exception as e:
            print(f"❌ Error loading CICIDS2017 test data: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Load new datasets test data
    try:
        X_test_new, y_test_new, new_features, new_scaler = load_test_data(
            ["cic_iot_2024", "cicapt_iiot"], sample_size=10000, min_samples_per_class=2000
        )
    except Exception as e:
        print(f"❌ Error loading new datasets test data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate models
    print("\n3. Evaluating models...")
    results = {}
    
    # Evaluate old models on CICIDS2017 test data
    print("\n   Evaluating old models on CICIDS2017 test data...")
    for model_name, model_info in old_models.items():
        try:
            X_aligned = align_features(
                X_test_old, old_features, model_info['feature_names'],
                old_scaler, model_info['scaler']
            )
            result = evaluate_model(
                model_info['model'], X_aligned, y_test_old,
                f"old_{model_name}", model_info['type']
            )
            if result:
                results[f"old_{model_name}"] = result
                print(f"   ✅ {model_name}: Accuracy = {result['accuracy']:.4f}")
        except Exception as e:
            print(f"   ❌ Error evaluating old {model_name}: {e}")
    
    # Evaluate new models on new datasets test data
    print("\n   Evaluating new models on new datasets test data...")
    for model_name, model_info in new_models.items():
        try:
            X_aligned = align_features(
                X_test_new, new_features, model_info['feature_names'],
                new_scaler, model_info['scaler']
            )
            result = evaluate_model(
                model_info['model'], X_aligned, y_test_new,
                f"new_{model_name}", model_info['type']
            )
            if result:
                results[f"new_{model_name}"] = result
                print(f"   ✅ {model_name}: Accuracy = {result['accuracy']:.4f}")
        except Exception as e:
            print(f"   ❌ Error evaluating new {model_name}: {e}")
    
    # Also evaluate old models on new test data for fair comparison
    print("\n   Evaluating old models on new datasets test data (for fair comparison)...")
    for model_name, model_info in old_models.items():
        try:
            X_aligned = align_features(
                X_test_new, new_features, model_info['feature_names'],
                new_scaler, model_info['scaler']
            )
            result = evaluate_model(
                model_info['model'], X_aligned, y_test_new,
                f"old_{model_name}_on_new_data", model_info['type']
            )
            if result:
                results[f"old_{model_name}_on_new_data"] = result
                print(f"   ✅ {model_name} on new data: Accuracy = {result['accuracy']:.4f}")
        except Exception as e:
            print(f"   ❌ Error evaluating old {model_name} on new data: {e}")
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    output_dir = Path("results/comparison_report")
    output_dir.mkdir(exist_ok=True)
    
    create_comparison_visualizations(results, output_dir)
    
    # Generate report
    print("\n5. Generating report...")
    report_path = generate_report(results, output_dir)
    
    # Save results to JSON
    results_json = {}
    for key, value in results.items():
        results_json[key] = {
            'accuracy': float(value.get('accuracy', 0)),
            'precision': float(value.get('precision', 0)),
            'recall': float(value.get('recall', 0)),
            'f1_score': float(value.get('f1_score', 0)),
            'roc_auc': float(value.get('roc_auc', 0)) if value.get('roc_auc') else None,
            'confusion_matrix': value.get('confusion_matrix').tolist() if value.get('confusion_matrix') is not None else None
        }
    
    json_path = output_dir / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✅ Saved results: {json_path}")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Report: {report_path}")
    print(f"📁 All files saved in: {output_dir}")
    print("\nGenerated files:")
    print("  - metrics_comparison.png")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")
    print("  - metrics_table.png")
    print("  - model_comparison_report_*.md")

if __name__ == "__main__":
    main()

