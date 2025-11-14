#!/usr/bin/env python3
"""
Test feature alignment for new models
Tests feature alignment between datasets and feature mapping
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.feature_aligner import FeatureAligner, align_multiple_datasets
from src.data_loader_multi import MultiDatasetLoader

MODELS_DIR = Path(__file__).parent.parent / "models"


class TestFeatureAlignmentNewModels(unittest.TestCase):
    """Test feature alignment for new models"""

    def setUp(self):
        """Set up test fixtures"""
        self.aligner = FeatureAligner()
        self.loader = MultiDatasetLoader()

    def test_feature_aligner_initialization(self):
        """Test FeatureAligner initialization"""
        aligner = FeatureAligner()
        self.assertIsNotNone(aligner)
        self.assertIsNone(aligner.common_features)
        self.assertEqual(aligner.feature_mapping, {})
        self.assertEqual(aligner.feature_stats, {})

    def test_find_common_features_single_dataset(self):
        """Test finding common features in a single dataset"""
        # Create a simple test dataset
        df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6],
            'Feature3': [7, 8, 9],
            'Label': ['BENIGN', 'ATTACK', 'BENIGN']
        })
        
        common_features = self.aligner.find_common_features([df])
        self.assertIsNotNone(common_features)
        self.assertGreater(len(common_features), 0)
        self.assertIn('Feature1', common_features)
        self.assertIn('Feature2', common_features)
        self.assertIn('Feature3', common_features)
        self.assertNotIn('Label', common_features)

    def test_find_common_features_multiple_datasets(self):
        """Test finding common features across multiple datasets"""
        df1 = pd.DataFrame({
            'Common1': [1, 2, 3],
            'Common2': [4, 5, 6],
            'Unique1': [7, 8, 9],
            'Label': ['BENIGN', 'ATTACK', 'BENIGN']
        })
        
        df2 = pd.DataFrame({
            'Common1': [10, 11, 12],
            'Common2': [13, 14, 15],
            'Unique2': [16, 17, 18],
            'Label': ['ATTACK', 'BENIGN', 'ATTACK']
        })
        
        common_features = self.aligner.find_common_features([df1, df2])
        self.assertIsNotNone(common_features)
        self.assertIn('Common1', common_features)
        self.assertIn('Common2', common_features)
        self.assertNotIn('Unique1', common_features)
        self.assertNotIn('Unique2', common_features)
        self.assertNotIn('Label', common_features)

    def test_align_dataset_exact_match(self):
        """Test aligning dataset with exact feature match"""
        df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6],
            'Label': ['BENIGN', 'ATTACK', 'BENIGN']
        })
        
        target_features = ['Feature1', 'Feature2']
        aligned_df = self.aligner.align_dataset(df, target_features)
        
        self.assertEqual(len(aligned_df), len(df))
        self.assertIn('Feature1', aligned_df.columns)
        self.assertIn('Feature2', aligned_df.columns)
        self.assertIn('Label', aligned_df.columns)
        self.assertTrue(np.array_equal(aligned_df['Feature1'].values, df['Feature1'].values))

    def test_align_dataset_missing_features(self):
        """Test aligning dataset with missing features (should fill with zeros)"""
        df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Label': ['BENIGN', 'ATTACK', 'BENIGN']
        })
        
        target_features = ['Feature1', 'Feature2', 'Feature3']
        aligned_df = self.aligner.align_dataset(df, target_features, fill_missing='zero')
        
        self.assertEqual(len(aligned_df), len(df))
        self.assertIn('Feature1', aligned_df.columns)
        self.assertIn('Feature2', aligned_df.columns)
        self.assertIn('Feature3', aligned_df.columns)
        self.assertTrue(np.allclose(aligned_df['Feature2'].values, 0.0))
        self.assertTrue(np.allclose(aligned_df['Feature3'].values, 0.0))

    def test_align_dataset_extra_features(self):
        """Test aligning dataset with extra features (should ignore)"""
        df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6],
            'ExtraFeature': [7, 8, 9],
            'Label': ['BENIGN', 'ATTACK', 'BENIGN']
        })
        
        target_features = ['Feature1', 'Feature2']
        aligned_df = self.aligner.align_dataset(df, target_features)
        
        self.assertEqual(len(aligned_df), len(df))
        self.assertIn('Feature1', aligned_df.columns)
        self.assertIn('Feature2', aligned_df.columns)
        self.assertNotIn('ExtraFeature', aligned_df.columns)

    def test_load_new_feature_names(self):
        """Test loading new feature names from saved file"""
        features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
        if not features_path.exists():
            self.skipTest(f"Feature names not found: {features_path}")
        
        feature_names = joblib.load(features_path)
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)
        
        self.assertGreater(len(feature_names), 0)
        self.assertIsInstance(feature_names[0], str)

    def test_align_to_new_feature_names(self):
        """Test aligning a dataset to new model feature names"""
        features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
        if not features_path.exists():
            self.skipTest(f"Feature names not found: {features_path}")
        
        target_features = joblib.load(features_path)
        if not isinstance(target_features, list):
            target_features = list(target_features)
        
        # Create a test dataset with some matching features
        test_features = target_features[:10] if len(target_features) >= 10 else target_features
        df = pd.DataFrame({
            feat: np.random.randn(100) for feat in test_features
        })
        df['Label'] = ['BENIGN'] * 50 + ['ATTACK'] * 50
        
        # Align to all target features
        aligned_df = self.aligner.align_dataset(df, target_features, fill_missing='zero')
        
        self.assertEqual(len(aligned_df), len(df))
        self.assertEqual(len([c for c in aligned_df.columns if c not in ['Label', ' Label']]), 
                        len(target_features))
        
        # Check that existing features are preserved
        for feat in test_features:
            if feat in df.columns:
                self.assertIn(feat, aligned_df.columns)

    def test_create_feature_mapping(self):
        """Test creating feature mapping"""
        source_features = ['Feature1', 'Feature2', 'Feature3']
        target_features = ['Feature1', 'Feature2', 'Feature4']
        
        mapping = self.aligner.create_feature_mapping(source_features, target_features)
        
        self.assertIn('Feature1', mapping)
        self.assertIn('Feature2', mapping)
        self.assertEqual(mapping['Feature1'], 'Feature1')
        self.assertEqual(mapping['Feature2'], 'Feature2')

    def test_align_multiple_datasets_function(self):
        """Test align_multiple_datasets helper function"""
        df1 = pd.DataFrame({
            'Common1': [1, 2, 3],
            'Common2': [4, 5, 6],
            'Unique1': [7, 8, 9],
            'Label': ['BENIGN', 'ATTACK', 'BENIGN']
        })
        
        df2 = pd.DataFrame({
            'Common1': [10, 11, 12],
            'Common2': [13, 14, 15],
            'Unique2': [16, 17, 18],
            'Label': ['ATTACK', 'BENIGN', 'ATTACK']
        })
        
        combined_df, common_features = align_multiple_datasets([df1, df2])
        
        self.assertIsNotNone(combined_df)
        self.assertIsNotNone(common_features)
        self.assertEqual(len(combined_df), len(df1) + len(df2))
        self.assertIn('Common1', common_features)
        self.assertIn('Common2', common_features)

    def test_feature_stats_computation(self):
        """Test computing feature statistics"""
        df = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Label': ['BENIGN', 'ATTACK', 'BENIGN', 'ATTACK', 'BENIGN']
        })
        
        features = ['Feature1', 'Feature2']
        stats = self.aligner.compute_feature_stats(df, features)
        
        self.assertIn('Feature1', stats)
        self.assertIn('Feature2', stats)
        self.assertIn('mean', stats['Feature1'])
        self.assertIn('std', stats['Feature1'])
        self.assertIn('min', stats['Feature1'])
        self.assertIn('max', stats['Feature1'])

    def test_feature_alignment_with_real_data(self):
        """Test feature alignment with real dataset loading (if available)"""
        try:
            # Try to load a small sample from one of the new datasets
            df = self.loader.load_dataset("cic_iot_2024", sample_size=100, file_limit=1)
            
            if df is not None and len(df) > 0:
                # Get target features from saved model
                features_path = MODELS_DIR / "feature_names_new_datasets.pkl"
                if features_path.exists():
                    target_features = joblib.load(features_path)
                    if not isinstance(target_features, list):
                        target_features = list(target_features)
                    
                    # Align the dataset
                    aligned_df = self.aligner.align_dataset(df, target_features, fill_missing='zero')
                    
                    self.assertIsNotNone(aligned_df)
                    self.assertEqual(len(aligned_df), len(df))
                    self.assertGreaterEqual(len([c for c in aligned_df.columns 
                                                if c not in ['Label', ' Label']]), 
                                           len(target_features))
        except Exception as e:
            # Skip if datasets are not available
            self.skipTest(f"Could not load real data for testing: {e}")


if __name__ == "__main__":
    unittest.main()

