# tests/test_attention_explainer.py
import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from models.attention_explainer import AttentionModule, AttentionExplainer


class TestAttentionModule:
    """Test the AttentionModule neural network"""

    def test_initialization(self):
        """Test that AttentionModule initializes correctly"""
        input_dim = 10
        module = AttentionModule(input_dim)

        assert module is not None
        assert isinstance(module, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass returns correct shapes"""
        input_dim = 10
        batch_size = 5

        module = AttentionModule(input_dim)
        x = torch.randn(batch_size, input_dim)

        weighted_x, attention_weights = module(x)

        assert weighted_x.shape == (batch_size, input_dim)
        assert attention_weights.shape == (batch_size, input_dim)

    def test_attention_weights_range(self):
        """Test that attention weights are in valid range [0, 1]"""
        input_dim = 10
        batch_size = 3

        module = AttentionModule(input_dim)
        x = torch.randn(batch_size, input_dim)

        _, attention_weights = module(x)

        assert torch.all(attention_weights >= 0)
        assert torch.all(attention_weights <= 1)


class TestAttentionExplainer:
    """Test the AttentionExplainer class"""

    @pytest.fixture
    def explainer(self):
        """Create a basic explainer instance"""
        feature_names = [f"feature_{i}" for i in range(10)]
        return AttentionExplainer(feature_names=feature_names)

    @pytest.fixture
    def sample_data(self):
        """Create sample input data"""
        return np.random.randn(5, 10)

    def test_initialization(self, explainer):
        """Test explainer initialization"""
        assert explainer is not None
        assert len(explainer.feature_names) == 10
        assert explainer.attention_module is None  # Lazy initialization

    def test_compute_feature_attention(self, explainer, sample_data):
        """Test attention weight computation"""
        attention_weights = explainer.compute_feature_attention(sample_data)

        assert attention_weights.shape == sample_data.shape
        assert np.all(attention_weights >= 0)
        # Softmax should make each row sum to approximately 1
        row_sums = np.sum(attention_weights, axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.01)

    def test_get_top_features(self, explainer, sample_data):
        """Test top features extraction"""
        attention_weights = explainer.compute_feature_attention(sample_data)
        top_features = explainer.get_top_features(attention_weights, top_k=3)

        assert len(top_features) == sample_data.shape[0]
        for features in top_features:
            assert len(features) == 3
            for name, score in features:
                assert isinstance(name, str)
                assert isinstance(score, float)
                assert score >= 0

    def test_generate_explanation(self, explainer, sample_data):
        """Test explanation generation"""
        predictions = np.array([0.8, 0.3, 0.9, 0.1, 0.5])
        explanations = explainer.generate_explanation(sample_data, predictions)

        assert len(explanations) == len(predictions)
        for i, exp in enumerate(explanations):
            assert 'prediction' in exp
            assert 'threat_level' in exp
            assert 'attention_weights' in exp
            assert 'top_features' in exp
            assert 'explanation' in exp

            assert exp['prediction'] == predictions[i]
            assert exp['threat_level'] in ['High', 'Medium', 'Low']

    def test_threat_level_classification(self, explainer, sample_data):
        """Test correct threat level classification"""
        predictions = np.array([0.85, 0.5, 0.2])
        explanations = explainer.generate_explanation(sample_data[:3], predictions)

        assert explanations[0]['threat_level'] == 'High'
        assert explanations[1]['threat_level'] == 'Medium'
        assert explanations[2]['threat_level'] == 'Low'

    def test_visualize_attention(self, explainer, sample_data):
        """Test visualization data generation"""
        attention_weights = explainer.compute_feature_attention(sample_data)
        viz_data = explainer.visualize_attention(attention_weights)

        assert len(viz_data) == sample_data.shape[0]
        for viz in viz_data:
            assert 'features' in viz
            assert 'weights' in viz
            assert len(viz['features']) == len(viz['weights'])
            assert len(viz['features']) <= 10  # Top 10

    def test_with_baseline_model(self, tmp_path):
        """Test with a mock baseline model"""
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        # Create and save a mock model
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        rf.fit(X_train, y_train)

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(rf, model_path)

        # Create explainer with model
        feature_names = [f"feature_{i}" for i in range(10)]
        explainer = AttentionExplainer(
            feature_names=feature_names,
            model_path=str(model_path)
        )

        assert explainer.baseline_model is not None

        # Test with baseline importance
        X_test = np.random.randn(3, 10)
        attention_weights = explainer.compute_feature_attention(
            X_test,
            use_baseline_importance=True
        )

        assert attention_weights.shape == X_test.shape

    def test_explanation_text_format(self, explainer, sample_data):
        """Test that explanation text is properly formatted"""
        predictions = np.array([0.9])
        explanations = explainer.generate_explanation(sample_data[:1], predictions)

        exp_text = explanations[0]['explanation']

        assert 'Threat Level:' in exp_text
        assert 'Key indicators:' in exp_text
        assert 'importance' in exp_text.lower()


def test_integration():
    """Integration test for the full pipeline"""
    # Create explainer
    feature_names = [f"feature_{i}" for i in range(10)]
    explainer = AttentionExplainer(feature_names=feature_names)

    # Create sample data
    X = np.random.randn(3, 10)
    predictions = np.array([0.8, 0.4, 0.1])

    # Generate explanations
    explanations = explainer.generate_explanation(X, predictions)

    # Validate
    assert len(explanations) == 3
    assert all('threat_level' in exp for exp in explanations)
    assert all('top_features' in exp for exp in explanations)
    assert all(len(exp['top_features']) > 0 for exp in explanations)

    print("✓ Integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
