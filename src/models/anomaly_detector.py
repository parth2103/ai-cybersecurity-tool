from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np


class AnomalyDetector:
    def __init__(self, contamination=0.1):
        """
        contamination: expected proportion of outliers in the dataset
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def train(self, X_normal):
        """Train on normal traffic data only"""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_normal)

        # Fit the model
        self.model.fit(X_scaled)

        return self

    def detect_anomalies(self, X):
        """
        Returns:
        - predictions: -1 for anomalies, 1 for normal
        - anomaly_scores: The lower, the more abnormal
        """
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)

        # Convert to binary (0: normal, 1: anomaly)
        binary_predictions = (predictions == -1).astype(int)

        return binary_predictions, anomaly_scores

    def get_anomaly_percentage(self, X):
        """Calculate percentage of anomalies in the dataset"""
        predictions, _ = self.detect_anomalies(X)
        anomaly_percentage = (predictions == 1).mean() * 100
        return anomaly_percentage
