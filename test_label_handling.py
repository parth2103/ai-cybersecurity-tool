#!/usr/bin/env python3
"""
Test script to verify label handling for new datasets
"""

import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent / "src"))

from data_loader_multi import MultiDatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_label_handling():
    """Test label handling for each dataset"""
    loader = MultiDatasetLoader()
    
    # Test datasets (excluding global_threats as it has no labels)
    test_datasets = ["cic_iot_2024", "cicapt_iiot", "cicids2017"]
    
    for dataset_key in test_datasets:
        print(f"\n{'=' * 80}")
        print(f"Testing: {dataset_key}")
        print(f"{'=' * 80}")
        
        try:
            # Load a small sample
            df = loader.load_dataset(dataset_key, sample_size=1000, file_limit=5)
            
            # Check label column
            label_col = loader.datasets[dataset_key].label_column
            if label_col in df.columns:
                print(f"\nLabel Column: {label_col}")
                print(f"Label Distribution:")
                print(df[label_col].value_counts())
                
                # Check if labels are standardized
                unique_labels = df[label_col].unique()
                print(f"\nUnique Labels: {unique_labels}")
                
                # Check for BENIGN/ATTACK format
                label_values = [str(v).upper() for v in unique_labels]
                has_benign = any("BENIGN" in v for v in label_values)
                has_attack = any("ATTACK" in v for v in label_values)
                
                if has_benign or has_attack:
                    print("✅ Labels are standardized (BENIGN/ATTACK format)")
                else:
                    print("⚠️  Labels may not be standardized")
            else:
                print(f"⚠️  Label column '{label_col}' not found in dataset")
                
        except Exception as e:
            logger.error(f"Error testing {dataset_key}: {e}", exc_info=True)

if __name__ == "__main__":
    test_label_handling()

