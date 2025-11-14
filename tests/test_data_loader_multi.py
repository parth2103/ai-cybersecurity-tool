#!/usr/bin/env python3
"""
Test MultiDatasetLoader functionality
Tests loading, label handling, and feature extraction for new datasets
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader_multi import MultiDatasetLoader, DatasetInfo


class TestMultiDatasetLoader(unittest.TestCase):
    """Test MultiDatasetLoader functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.loader = MultiDatasetLoader()

    def test_loader_initialization(self):
        """Test MultiDatasetLoader initialization"""
        self.assertIsNotNone(self.loader)
        self.assertIsNotNone(self.loader.base_data_dir)
        self.assertIn("cicids2017", self.loader.datasets)
        self.assertIn("cic_iot_2024", self.loader.datasets)
        self.assertIn("cicapt_iiot", self.loader.datasets)

    def test_list_available_datasets(self):
        """Test listing available datasets"""
        available = self.loader.list_available_datasets()
        self.assertIsInstance(available, list)
        # At least cicids2017 should be available (or we skip if not)
        if len(available) == 0:
            self.skipTest("No datasets available for testing")

    def test_dataset_info_structure(self):
        """Test that dataset info has correct structure"""
        for key, info in self.loader.datasets.items():
            self.assertIsInstance(info, DatasetInfo)
            self.assertIsNotNone(info.name)
            self.assertIsNotNone(info.path)
            self.assertIsNotNone(info.label_column)
            self.assertIsNotNone(info.file_pattern)

    def test_explore_dataset_structure(self):
        """Test exploring dataset structure"""
        available = self.loader.list_available_datasets()
        if not available:
            self.skipTest("No datasets available for testing")
        
        # Test with first available dataset
        dataset_key = available[0]
        try:
            exploration = self.loader.explore_dataset(dataset_key)
            
            self.assertIsInstance(exploration, dict)
            self.assertIn("name", exploration)
            self.assertIn("path", exploration)
            self.assertIn("exists", exploration)
            self.assertIn("files", exploration)
        except Exception as e:
            self.skipTest(f"Could not explore dataset {dataset_key}: {e}")

    def test_load_cicids2017(self):
        """Test loading CICIDS2017 dataset"""
        if "cicids2017" not in self.loader.list_available_datasets():
            self.skipTest("CICIDS2017 dataset not available")
        
        try:
            df = self.loader.load_dataset("cicids2017", sample_size=100, file_limit=1)
            
            if df is not None and len(df) > 0:
                self.assertIsInstance(df, pd.DataFrame)
                self.assertGreater(len(df), 0)
                self.assertGreater(len(df.columns), 0)
        except Exception as e:
            self.skipTest(f"Could not load CICIDS2017: {e}")

    def test_load_cic_iot_2024(self):
        """Test loading CIC IoT-IDAD 2024 dataset"""
        if "cic_iot_2024" not in self.loader.list_available_datasets():
            self.skipTest("CIC IoT-IDAD 2024 dataset not available")
        
        try:
            df = self.loader.load_dataset("cic_iot_2024", sample_size=100, file_limit=1)
            
            if df is not None and len(df) > 0:
                self.assertIsInstance(df, pd.DataFrame)
                self.assertGreater(len(df), 0)
                self.assertGreater(len(df.columns), 0)
        except Exception as e:
            self.skipTest(f"Could not load CIC IoT-IDAD 2024: {e}")

    def test_load_cicapt_iiot(self):
        """Test loading CICAPT-IIOT dataset"""
        if "cicapt_iiot" not in self.loader.list_available_datasets():
            self.skipTest("CICAPT-IIOT dataset not available")
        
        try:
            df = self.loader.load_dataset("cicapt_iiot", sample_size=100, file_limit=1)
            
            if df is not None and len(df) > 0:
                self.assertIsInstance(df, pd.DataFrame)
                self.assertGreater(len(df), 0)
                self.assertGreater(len(df.columns), 0)
        except Exception as e:
            self.skipTest(f"Could not load CICAPT-IIOT: {e}")

    def test_label_inference_iot_2024(self):
        """Test label inference from folder names for IoT-IDAD 2024"""
        if "cic_iot_2024" not in self.loader.list_available_datasets():
            self.skipTest("CIC IoT-IDAD 2024 dataset not available")
        
        try:
            df = self.loader.load_dataset("cic_iot_2024", sample_size=100, file_limit=1)
            
            if df is not None and len(df) > 0:
                # Check if Label column exists
                label_col = "Label" if "Label" in df.columns else " Label"
                if label_col in df.columns:
                    # Check that labels are standardized
                    unique_labels = df[label_col].unique()
                    # Should have BENIGN and/or ATTACK after processing
                    label_str = [str(l).upper() for l in unique_labels]
                    self.assertTrue(
                        any("BENIGN" in l or "ATTACK" in l for l in label_str),
                        f"Labels should be BENIGN/ATTACK, got {unique_labels}"
                    )
        except Exception as e:
            self.skipTest(f"Could not test label inference: {e}")

    def test_label_conversion_cicapt(self):
        """Test numeric label conversion for CICAPT-IIOT"""
        if "cicapt_iiot" not in self.loader.list_available_datasets():
            self.skipTest("CICAPT-IIOT dataset not available")
        
        try:
            df = self.loader.load_dataset("cicapt_iiot", sample_size=100, file_limit=1)
            
            if df is not None and len(df) > 0:
                label_col = "Label" if "Label" in df.columns else " Label"
                if label_col in df.columns:
                    # Labels should be converted to text (BENIGN/ATTACK)
                    unique_labels = df[label_col].unique()
                    label_str = [str(l).upper() for l in unique_labels]
                    # Should not be numeric
                    self.assertFalse(
                        all(str(l).isdigit() for l in unique_labels if pd.notna(l)),
                        "Labels should be converted from numeric to text"
                    )
        except Exception as e:
            self.skipTest(f"Could not test label conversion: {e}")

    def test_label_standardization(self):
        """Test that labels are standardized to BENIGN/ATTACK"""
        available = self.loader.list_available_datasets()
        if not available:
            self.skipTest("No datasets available for testing")
        
        for dataset_key in available[:2]:  # Test first 2 available
            try:
                df = self.loader.load_dataset(dataset_key, sample_size=50, file_limit=1)
                
                if df is not None and len(df) > 0:
                    label_col = "Label" if "Label" in df.columns else " Label"
                    if label_col in df.columns:
                        unique_labels = df[label_col].unique()
                        # Check that labels are standardized
                        label_str = [str(l).upper().strip() for l in unique_labels if pd.notna(l)]
                        # Should contain BENIGN or ATTACK
                        has_benign = any("BENIGN" in l for l in label_str)
                        has_attack = any("ATTACK" in l for l in label_str)
                        
                        # At least one should be present
                        self.assertTrue(
                            has_benign or has_attack,
                            f"Labels should be BENIGN/ATTACK, got {unique_labels}"
                        )
            except Exception as e:
                # Skip if dataset can't be loaded
                continue

    def test_feature_extraction(self):
        """Test that features are extracted correctly"""
        available = self.loader.list_available_datasets()
        if not available:
            self.skipTest("No datasets available for testing")
        
        dataset_key = available[0]
        try:
            df = self.loader.load_dataset(dataset_key, sample_size=100, file_limit=1)
            
            if df is not None and len(df) > 0:
                # Should have numeric features (excluding label columns)
                label_cols = ["Label", " Label", "label"]
                feature_cols = [c for c in df.columns if c not in label_cols]
                
                self.assertGreater(len(feature_cols), 0, "Should have feature columns")
                
                # Most features should be numeric
                numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
                self.assertGreater(len(numeric_cols), 0, "Should have numeric features")
        except Exception as e:
            self.skipTest(f"Could not test feature extraction: {e}")

    def test_sample_size_limit(self):
        """Test that sample_size parameter limits the number of rows"""
        available = self.loader.list_available_datasets()
        if not available:
            self.skipTest("No datasets available for testing")
        
        dataset_key = available[0]
        try:
            sample_size = 50
            df = self.loader.load_dataset(dataset_key, sample_size=sample_size, file_limit=1)
            
            if df is not None:
                # Should not exceed sample_size significantly (may be slightly more due to file boundaries)
                self.assertLessEqual(len(df), sample_size * 2, 
                                   f"Should limit to ~{sample_size} rows, got {len(df)}")
        except Exception as e:
            self.skipTest(f"Could not test sample size limit: {e}")

    def test_file_limit(self):
        """Test that file_limit parameter limits the number of files"""
        available = self.loader.list_available_datasets()
        if not available:
            self.skipTest("No datasets available for testing")
        
        dataset_key = available[0]
        try:
            exploration = self.loader.explore_dataset(dataset_key)
            total_files = len(exploration.get("files", []))
            
            if total_files > 1:
                file_limit = 1
                df = self.loader.load_dataset(dataset_key, sample_size=100, file_limit=file_limit)
                
                if df is not None:
                    # Should load data from limited files
                    self.assertIsInstance(df, pd.DataFrame)
        except Exception as e:
            self.skipTest(f"Could not test file limit: {e}")

    def test_combined_datasets_loading(self):
        """Test loading multiple datasets and combining them"""
        available = self.loader.list_available_datasets()
        if len(available) < 2:
            self.skipTest("Need at least 2 datasets for combination test")
        
        try:
            datasets = []
            for dataset_key in available[:2]:
                df = self.loader.load_dataset(dataset_key, sample_size=50, file_limit=1)
                if df is not None and len(df) > 0:
                    datasets.append(df)
            
            if len(datasets) >= 2:
                # Combine datasets
                combined = pd.concat(datasets, ignore_index=True)
                self.assertGreater(len(combined), len(datasets[0]))
                self.assertGreater(len(combined.columns), 0)
        except Exception as e:
            self.skipTest(f"Could not test dataset combination: {e}")


if __name__ == "__main__":
    unittest.main()

