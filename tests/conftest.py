"""
Pytest configuration and fixtures for AI Cybersecurity Tool tests
"""

import pytest
import requests
import time
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def api_available():
    """Check if API server is available"""
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="session")
def api_base_url():
    """API base URL"""
    return "http://localhost:5001"


@pytest.fixture
def sample_features():
    """Sample feature data for testing"""
    return {
        "Destination Port": 80,
        "Flow Duration": 1000,
        "Total Fwd Packets": 100,
        "Total Backward Packets": 100,
        "Total Length of Fwd Packets": 10000,
        "Total Length of Bwd Packets": 10000,
        "Fwd Packet Length Max": 1500,
        "Fwd Packet Length Min": 64,
        "Fwd Packet Length Mean": 100,
        "Fwd Packet Length Std": 50,
        "Bwd Packet Length Max": 1500,
        "Bwd Packet Length Min": 64,
        "Bwd Packet Length Mean": 100,
        "Bwd Packet Length Std": 50,
        "Flow Bytes/s": 2000,
        "Flow Packets/s": 20,
        "Flow IAT Mean": 50,
        "Flow IAT Std": 10,
        "Flow IAT Max": 100,
        "Flow IAT Min": 10,
    }


@pytest.fixture
def sample_request_data(sample_features):
    """Sample request data for API testing"""
    return {
        "features": sample_features,
        "source_ip": "192.168.1.100",
        "attack_type": "test",
    }


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0])
    model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1]])
    return model


@pytest.fixture
def sample_training_data():
    """Sample training data for testing"""
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "Feature_1": np.random.randn(100),
            "Feature_2": np.random.randn(100),
            "Feature_3": np.random.randn(100),
            "Label": np.random.choice(["BENIGN", "DDoS"], 100),
        }
    )


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "api: Tests requiring API server")
    config.addinivalue_line("markers", "docker: Tests requiring Docker")


# Skip integration tests if API is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available services"""
    api_available = False
    try:
        response = requests.get("http://localhost:5001/health", timeout=2)
        api_available = response.status_code == 200
    except:
        pass

    if not api_available:
        skip_api = pytest.mark.skip(reason="API server not available")
        for item in items:
            if "integration" in item.keywords or "api" in item.keywords:
                item.add_marker(skip_api)
