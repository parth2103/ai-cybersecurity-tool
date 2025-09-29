import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, y_proba=None, model_name="Model"):
        """Comprehensive model evaluation"""
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        if y_proba is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_proba)
            
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Store results
        self.results[model_name] = results
        
        return results
    
    def plot_confusion_matrix(self, cm, model_name, class_names=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{model_name}.png')
        plt.close()  # Close instead of show
    
    def compare_models(self):
        """Create comparison table of all evaluated models"""
        if not self.results:
            print("No models evaluated yet")
            return
            
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.drop('confusion_matrix', axis=1, errors='ignore')
        
        # Sort by F1 score
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(comparison_df.to_string())
        
        # Save to CSV
        comparison_df.to_csv('results/model_comparison.csv')
        
        return comparison_df
    
    def plot_model_comparison(self):
        """Visual comparison of models"""
        df = pd.DataFrame(self.results).T
        df = df.drop('confusion_matrix', axis=1, errors='ignore')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            df[metric].plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f'{metric.upper()} Comparison')
            axes[idx].set_xlabel('Model')
            axes[idx].set_ylabel(metric)
            axes[idx].set_ylim([0, 1])
            
        plt.tight_layout()
        plt.savefig('results/model_comparison.png')
        plt.close()  # Close instead of show
