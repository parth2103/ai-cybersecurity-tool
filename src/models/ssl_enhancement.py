"""
Self-Supervised Learning Enhancement Module for Cybersecurity Detection

This module implements SimCLR-style contrastive learning for network traffic data
to enhance feature representations without replacing existing models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkTrafficDataset(Dataset):
    """Dataset class for network traffic data with augmentation support."""
    
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None, 
                 scaler: Optional[StandardScaler] = None, augment: bool = True):
        """
        Args:
            data: Network traffic features (n_samples, n_features)
            labels: Optional labels for supervised learning
            scaler: Optional scaler for normalization
            augment: Whether to apply data augmentation
        """
        self.data = torch.FloatTensor(data)
        self.labels = labels
        self.scaler = scaler
        self.augment = augment
        
        # Normalize data if scaler is provided
        if self.scaler is not None:
            self.data = torch.FloatTensor(self.scaler.transform(data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        
        if self.augment:
            # Create two augmented views for contrastive learning
            x1 = self._augment_features(x.clone())
            x2 = self._augment_features(x.clone())
            return x1, x2
        
        return x
    
    def _augment_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply network-specific data augmentation."""
        # Feature dropout (randomly zero out some features)
        if random.random() < 0.3:
            dropout_mask = torch.rand_like(x) > 0.1
            x = x * dropout_mask
        
        # Gaussian noise injection
        if random.random() < 0.4:
            noise_std = 0.05 * torch.std(x)
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        
        # Feature scaling (simulate different network conditions)
        if random.random() < 0.3:
            scale_factor = torch.rand(1) * 0.4 + 0.8  # Random value between 0.8 and 1.2
            x = x * scale_factor
        
        # Temporal shift simulation (for time-series-like features)
        if random.random() < 0.2:
            shift_amount = random.uniform(-0.1, 0.1)
            x = x + shift_amount
        
        return x


class SSLEncoder(nn.Module):
    """Self-supervised learning encoder for network traffic features."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], 
                 output_dim: int = 32, dropout: float = 0.2):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(SSLEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output projection layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        """Forward pass through encoder and projector."""
        # Encode features
        features = self.encoder(x)
        
        # Project to contrastive learning space
        projections = self.projector(features)
        
        return features, projections


class SSLEnhancement:
    """Self-supervised learning enhancement for cybersecurity detection."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], 
                 output_dim: int = 32, learning_rate: float = 0.001, 
                 temperature: float = 0.1, device: str = 'auto'):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions for encoder
            output_dim: Output embedding dimension
            learning_rate: Learning rate for optimizer
            temperature: Temperature parameter for contrastive loss
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize encoder
        self.encoder = SSLEncoder(input_dim, hidden_dims, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        
        # Data scaler
        self.scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': []
        }
    
    def contrastive_loss(self, projections1: torch.Tensor, projections2: torch.Tensor) -> torch.Tensor:
        """Compute SimCLR-style contrastive loss."""
        batch_size = projections1.size(0)
        
        # Normalize projections
        projections1 = F.normalize(projections1, dim=1)
        projections2 = F.normalize(projections2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections1, projections2.T) / self.temperature
        
        # Create labels for positive pairs (diagonal elements)
        labels = torch.arange(batch_size).to(self.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def pretrain(self, X: np.ndarray, epochs: int = 100, batch_size: int = 256, 
                 validation_split: float = 0.2, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Pretrain the SSL encoder using unlabeled data.
        
        Args:
            X: Unlabeled network traffic data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            save_path: Optional path to save the trained encoder
        
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Starting SSL pretraining with {len(X)} samples")
        
        # Clean data - handle infinity and NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Split data
        X_train, X_val = train_test_split(X, test_size=validation_split, random_state=42)
        
        # Fit scaler on training data
        self.scaler.fit(X_train)
        
        # Create datasets
        train_dataset = NetworkTrafficDataset(X_train, scaler=self.scaler, augment=True)
        val_dataset = NetworkTrafficDataset(X_val, scaler=self.scaler, augment=True)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.encoder.train()
            train_loss = 0.0
            
            for batch_x1, batch_x2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_x1, batch_x2 = batch_x1.to(self.device), batch_x2.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                _, projections1 = self.encoder(batch_x1)
                _, projections2 = self.encoder(batch_x2)
                
                # Compute loss
                loss = self.contrastive_loss(projections1, projections2)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.encoder.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x1, batch_x2 in val_loader:
                    batch_x1, batch_x2 = batch_x1.to(self.device), batch_x2.to(self.device)
                    
                    _, projections1 = self.encoder(batch_x1)
                    _, projections2 = self.encoder(batch_x2)
                    
                    loss = self.contrastive_loss(projections1, projections2)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Store training history
            self.training_history['loss'].append(avg_train_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save encoder if path provided
        if save_path:
            self.save_encoder(save_path)
            logger.info(f"Encoder saved to {save_path}")
        
        return {
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def encode_features(self, X: np.ndarray) -> np.ndarray:
        """
        Encode features using the trained SSL encoder.
        
        Args:
            X: Input features to encode
        
        Returns:
            Encoded features
        """
        self.encoder.eval()
        
        # Clean data - handle infinity and NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize input
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            features, _ = self.encoder(X_tensor)
        
        return features.cpu().numpy()
    
    def save_encoder(self, path: str):
        """Save the trained encoder and scaler."""
        save_data = {
            'encoder_state_dict': self.encoder.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'training_history': self.training_history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_data, path)
    
    @classmethod
    def load_encoder(cls, path: str, device: str = 'auto'):
        """Load a trained encoder."""
        save_data = joblib.load(path)
        
        # Create instance
        instance = cls(
            input_dim=save_data['input_dim'],
            output_dim=save_data['output_dim'],
            device=device
        )
        
        # Load state
        instance.encoder.load_state_dict(save_data['encoder_state_dict'])
        instance.scaler = save_data['scaler']
        instance.training_history = save_data.get('training_history', {})
        
        return instance


def create_ssl_enhanced_features(original_features: np.ndarray, 
                                ssl_encoder_path: str) -> np.ndarray:
    """
    Create SSL-enhanced features by combining original and SSL-encoded features.
    
    Args:
        original_features: Original network traffic features
        ssl_encoder_path: Path to trained SSL encoder
    
    Returns:
        Combined features (original + SSL-encoded)
    """
    # Load SSL encoder
    ssl_enhancement = SSLEnhancement.load_encoder(ssl_encoder_path)
    
    # Generate SSL features
    ssl_features = ssl_enhancement.encode_features(original_features)
    
    # Combine original and SSL features
    enhanced_features = np.concatenate([original_features, ssl_features], axis=1)
    
    logger.info(f"Enhanced features shape: {enhanced_features.shape} "
                f"(original: {original_features.shape[1]}, SSL: {ssl_features.shape[1]})")
    
    return enhanced_features


if __name__ == "__main__":
    # Example usage
    from data_loader import CICIDSDataLoader
    
    # Load data
    loader = CICIDSDataLoader()
    df = loader.load_friday_data(sample_size=5000)
    
    # Prepare features (exclude labels for SSL)
    feature_cols = [col for col in df.columns if col not in ['Label', ' Label']]
    X = df[feature_cols].values
    
    # Initialize and train SSL encoder
    ssl = SSLEnhancement(input_dim=X.shape[1])
    
    # Pretrain on unlabeled data
    metrics = ssl.pretrain(X, epochs=50, batch_size=128, 
                          save_path='models/ssl_encoder.pkl')
    
    print(f"SSL Training completed: {metrics}")
    
    # Test feature enhancement
    enhanced_features = create_ssl_enhanced_features(X[:100], 'models/ssl_encoder.pkl')
    print(f"Enhanced features shape: {enhanced_features.shape}")
