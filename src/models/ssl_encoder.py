#!/usr/bin/env python3
"""
SSL Encoder for AI Cybersecurity Tool
Implements a PyTorch-based SSL encoder for feature transformation
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
from typing import Union, Optional
from pathlib import Path


class SSLEncoder(nn.Module):
    """
    Self-Supervised Learning Encoder for feature transformation
    Matches the exact architecture from the saved model
    """
    
    def __init__(self, input_dim: int = 78, output_dim: int = 32):
        super(SSLEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Encoder layers (exact match with saved model)
        self.encoder = nn.ModuleDict({
            '0': nn.Linear(input_dim, 256),      # encoder.0: 78 -> 256
            '1': nn.BatchNorm1d(256),            # encoder.1: BatchNorm
            '4': nn.Linear(256, 128),            # encoder.4: 256 -> 128
            '5': nn.BatchNorm1d(128),            # encoder.5: BatchNorm
            '8': nn.Linear(128, 64),             # encoder.8: 128 -> 64
            '9': nn.BatchNorm1d(64),             # encoder.9: BatchNorm
            '12': nn.Linear(64, output_dim)      # encoder.12: 64 -> 32
        })
        
        # Projector (for SSL training)
        self.projector = nn.ModuleDict({
            '0': nn.Linear(output_dim, 32),      # projector.0: 32 -> 32
            '2': nn.Linear(32, 32)               # projector.2: 32 -> 32
        })
    
    def forward(self, x):
        """Forward pass through the encoder"""
        # Forward through encoder layers (in order)
        x = self.encoder['0'](x)      # Linear: 78 -> 256
        x = self.encoder['1'](x)      # BatchNorm
        x = torch.relu(x)             # ReLU
        x = self.encoder['4'](x)      # Linear: 256 -> 128
        x = self.encoder['5'](x)      # BatchNorm
        x = torch.relu(x)             # ReLU
        x = self.encoder['8'](x)      # Linear: 128 -> 64
        x = self.encoder['9'](x)      # BatchNorm
        x = torch.relu(x)             # ReLU
        encoded = self.encoder['12'](x)  # Linear: 64 -> 32
        
        # Forward through projector
        x = self.projector['0'](encoded)  # Linear: 32 -> 32
        x = torch.relu(x)                 # ReLU
        projected = self.projector['2'](x)  # Linear: 32 -> 32
        
        return encoded, projected
    
    def encode(self, x):
        """Get encoded representation only"""
        # Forward through encoder layers (in order)
        x = self.encoder['0'](x)      # Linear: 78 -> 256
        x = self.encoder['1'](x)      # BatchNorm
        x = torch.relu(x)             # ReLU
        x = self.encoder['4'](x)      # Linear: 256 -> 128
        x = self.encoder['5'](x)      # BatchNorm
        x = torch.relu(x)             # ReLU
        x = self.encoder['8'](x)      # Linear: 128 -> 64
        x = self.encoder['9'](x)      # BatchNorm
        x = torch.relu(x)             # ReLU
        encoded = self.encoder['12'](x)  # Linear: 64 -> 32
        
        return encoded


class SSLEncoderWrapper:
    """
    Wrapper class to make SSL encoder compatible with sklearn-style interface
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.encoder = None
        self.scaler = None
        self.input_dim = None
        self.output_dim = None
        self.device = torch.device('cpu')  # Use CPU for compatibility
        
        if model_path and Path(model_path).exists():
            self.load_from_file(model_path)
    
    def load_from_file(self, model_path: str):
        """Load SSL encoder from saved file"""
        try:
            # Load the saved data
            ssl_data = joblib.load(model_path)
            
            # Extract components
            self.input_dim = ssl_data['input_dim']
            self.output_dim = ssl_data['output_dim']
            self.scaler = ssl_data['scaler']
            state_dict = ssl_data['encoder_state_dict']
            
            # Create the encoder model
            self.encoder = SSLEncoder(self.input_dim, self.output_dim)
            
            # Load the state dict
            self.encoder.load_state_dict(state_dict)
            self.encoder.eval()  # Set to evaluation mode
            
            print(f"SSL Encoder loaded: {self.input_dim} -> {self.output_dim}")
            
        except Exception as e:
            print(f"Failed to load SSL encoder: {e}")
            raise
    
    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transform features using SSL encoder
        Compatible with sklearn-style interface
        """
        if self.encoder is None:
            raise ValueError("SSL encoder not loaded")
        
        # Convert to numpy array if needed
        if isinstance(X, list):
            X = np.array(X)
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get encoded features
        with torch.no_grad():
            encoded = self.encoder.encode(X_tensor)
            encoded_np = encoded.cpu().numpy()
        
        return encoded_np
    
    def fit(self, X, y=None):
        """Dummy fit method for sklearn compatibility"""
        return self
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.transform(X)


def load_ssl_encoder(model_path: str = "models/ssl_encoder.pkl") -> SSLEncoderWrapper:
    """
    Load SSL encoder from file
    """
    return SSLEncoderWrapper(model_path)


# Test function
if __name__ == "__main__":
    # Test the SSL encoder
    try:
        encoder = load_ssl_encoder()
        
        # Create test data
        test_data = np.random.randn(5, 78)  # 5 samples, 78 features
        
        # Transform
        encoded = encoder.transform(test_data)
        print(f"Original shape: {test_data.shape}")
        print(f"Encoded shape: {encoded.shape}")
        print("SSL Encoder test successful!")
        
    except Exception as e:
        print(f"SSL Encoder test failed: {e}")
