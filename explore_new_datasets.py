#!/usr/bin/env python3
"""
Dataset Exploration Script

Explores new cybersecurity datasets to understand their structure,
features, and compatibility with existing models.
"""

import sys
from pathlib import Path
import json
import logging
import numpy as np
from src.data_loader_multi import MultiDatasetLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def explore_all_datasets():
    """Explore all available datasets"""
    loader = MultiDatasetLoader()
    
    # List available datasets
    print("=" * 80)
    print("DATASET EXPLORATION")
    print("=" * 80)
    
    available = loader.list_available_datasets()
    print(f"\nAvailable datasets: {len(available)}")
    for key in available:
        info = loader.get_dataset_info(key)
        print(f"  - {key}: {info.name} ({info.path})")
    
    if not available:
        print("\n⚠️  No datasets found. Please ensure datasets are downloaded and placed in:")
        print("   - data/cic_iot_idad_2024/Dataset/")
        print("   - data/cicapt_iiot/")
        print("   - data/global_cybersecurity_threats/")
        return
    
    # Explore each dataset
    exploration_results = {}
    
    for key in available:
        print(f"\n{'=' * 80}")
        print(f"Exploring: {key}")
        print(f"{'=' * 80}")
        
        try:
            info = loader.explore_dataset(key)
            exploration_results[key] = info
            
            # Print summary
            print(f"\nDataset: {info['name']}")
            print(f"Path: {info['path']}")
            print(f"Exists: {info['exists']}")
            
            if info['exists']:
                print(f"Files found: {len(info['files'])}")
                if info['files']:
                    print(f"Sample file: {Path(info['files'][0]).name}")
                
                if info.get('columns'):
                    print(f"Total columns: {len(info['columns'])}")
                    print(f"Numeric features: {info.get('num_features', 0)}")
                
                if info.get('label_column'):
                    print(f"Label column: {info['label_column']}")
                    if info.get('label_values'):
                        print(f"Label classes: {len(info['label_values'])}")
                        print("Label distribution:")
                        for label, count in list(info['label_values'].items())[:10]:
                            print(f"  {label}: {count}")
                
                if info.get('sample_rows'):
                    print(f"Sample rows loaded: {info['sample_rows']}")
                
                if info.get('missing_values'):
                    missing_count = sum(1 for v in info['missing_values'].values() if v > 0)
                    if missing_count > 0:
                        print(f"Columns with missing values: {missing_count}")
                        # Show top missing columns
                        missing_sorted = sorted(
                            info['missing_percentage'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                        print("Top missing columns:")
                        for col, pct in missing_sorted:
                            if pct > 0:
                                print(f"  {col}: {pct:.2f}%")
            
        except Exception as e:
            logger.error(f"Error exploring {key}: {e}", exc_info=True)
            exploration_results[key] = {"error": str(e)}
    
    # Save exploration results
    results_path = Path("results/dataset_exploration.json")
    results_path.parent.mkdir(exist_ok=True)
    
    # Convert Path objects and numpy types to strings for JSON serialization
    def make_serializable(obj):
        """Recursively convert objects to JSON-serializable types"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, 'dtype'):  # numpy dtype
            return str(obj)
        else:
            return obj
    
    serializable_results = {}
    for key, value in exploration_results.items():
        serializable_results[key] = make_serializable(value)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Exploration results saved to: {results_path}")
    print(f"{'=' * 80}")
    
    return exploration_results


def compare_datasets():
    """Compare features across datasets"""
    loader = MultiDatasetLoader()
    available = loader.list_available_datasets()
    
    if len(available) < 2:
        print("Need at least 2 datasets to compare")
        return
    
    print("\n" + "=" * 80)
    print("FEATURE COMPARISON")
    print("=" * 80)
    
    # Load sample from each dataset
    datasets_info = {}
    for key in available:
        try:
            info = loader.explore_dataset(key)
            if info.get('numeric_columns'):
                datasets_info[key] = set(info['numeric_columns'])
                print(f"\n{key}: {len(datasets_info[key])} numeric features")
        except Exception as e:
            logger.error(f"Error comparing {key}: {e}")
    
    if len(datasets_info) < 2:
        print("Could not compare datasets")
        return
    
    # Find common features
    feature_sets = list(datasets_info.values())
    common_features = set.intersection(*feature_sets)
    
    print(f"\nCommon features across all datasets: {len(common_features)}")
    if common_features:
        print("Sample common features:")
        for feat in list(common_features)[:10]:
            print(f"  - {feat}")
    
    # Find features in majority of datasets
    all_features = set.union(*feature_sets)
    feature_counts = {}
    for feat in all_features:
        count = sum(1 for fs in feature_sets if feat in fs)
        feature_counts[feat] = count
    
    majority_threshold = len(datasets_info) * 0.5
    majority_features = {
        feat for feat, count in feature_counts.items()
        if count >= majority_threshold
    }
    
    print(f"\nFeatures in ≥50% of datasets: {len(majority_features)}")
    
    # Dataset-specific features
    for key, features in datasets_info.items():
        unique = features - set.union(*[fs for k, fs in datasets_info.items() if k != key])
        print(f"\n{key} unique features: {len(unique)}")
        if unique:
            print("Sample unique features:")
            for feat in list(unique)[:5]:
                print(f"  - {feat}")


if __name__ == "__main__":
    print("Dataset Exploration Tool")
    print("=" * 80)
    
    # Explore all datasets
    results = explore_all_datasets()
    
    # Compare datasets
    if results:
        compare_datasets()
    
    print("\n✅ Exploration complete!")
    print("\nNext steps:")
    print("1. Review exploration results in results/dataset_exploration.json")
    print("2. Update dataset configurations if label columns differ")
    print("3. Run: python train_on_new_datasets.py")

