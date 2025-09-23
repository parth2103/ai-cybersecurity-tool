import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
import numpy as np

class XGBoostDetector:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',  # Faster on M1
            n_jobs=-1
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train with optional early stopping"""
        eval_set = [(X_train, y_train)]
        
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        return self.model
    
    def predict_proba(self, X):
        """Get probability scores for threat detection"""
        return self.model.predict_proba(X)[:, 1]
    
    def optimize_threshold(self, X_val, y_val):
        """Find optimal threshold for classification"""
        probas = self.predict_proba(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, probas)
        
        # Find threshold with best F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[best_threshold_idx]
        
        return self.optimal_threshold
