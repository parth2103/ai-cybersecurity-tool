"""
Feature Alignment Utility

Handles feature alignment and unification across different datasets
with potentially different feature sets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class FeatureAligner:
    """Align features across different datasets"""
    
    def __init__(self):
        self.common_features: Optional[List[str]] = None
        self.feature_mapping: Dict[str, Dict[str, str]] = {}
        self.feature_stats: Dict[str, Dict] = {}
    
    def find_common_features(self, datasets: List[pd.DataFrame]) -> List[str]:
        """
        Find common features across multiple datasets
        
        Args:
            datasets: List of dataframes from different datasets
            
        Returns:
            List of common feature names
        """
        if not datasets:
            return []
        
        # Get all numeric columns from each dataset (excluding metadata columns)
        metadata_cols = ["_dataset_source", "Label", " Label", "label"]
        
        feature_sets = []
        for i, df in enumerate(datasets):
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude metadata and label columns
            features = [col for col in numeric_cols if col not in metadata_cols]
            feature_sets.append(set(features))
            logger.info(f"Dataset {i}: {len(features)} numeric features")
        
        # Find intersection of all feature sets
        common_features = set.intersection(*feature_sets)
        logger.info(f"Common features: {len(common_features)}")
        
        # Also find features that exist in majority of datasets (for flexible alignment)
        all_features = set.union(*feature_sets)
        feature_counts = {feat: sum(1 for fs in feature_sets if feat in fs) for feat in all_features}
        majority_threshold = len(datasets) * 0.5  # At least 50% of datasets
        
        majority_features = {
            feat for feat, count in feature_counts.items()
            if count >= majority_threshold and feat not in metadata_cols
        }
        
        logger.info(f"Majority features (≥50% datasets): {len(majority_features)}")
        
        # Prefer common features, but include majority features if common is too small
        if len(common_features) < 10:
            logger.warning(f"Too few common features ({len(common_features)}), using majority features")
            self.common_features = sorted(majority_features)
        else:
            self.common_features = sorted(common_features)
        
        return self.common_features
    
    def align_dataset(
        self,
        df: pd.DataFrame,
        target_features: List[str],
        fill_missing: str = "zero"
    ) -> pd.DataFrame:
        """
        Align a dataset to target features
        
        Args:
            df: Input dataframe
            target_features: List of target feature names
            fill_missing: How to fill missing features ("zero", "mean", "median")
            
        Returns:
            Aligned dataframe with target features
        """
        # Get numeric columns (excluding metadata)
        metadata_cols = ["_dataset_source", "Label", " Label", "label"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in numeric_cols if col not in metadata_cols]
        
        # Create aligned dataframe
        aligned_df = pd.DataFrame(index=df.index)
        
        # Copy metadata columns
        for col in metadata_cols:
            if col in df.columns:
                aligned_df[col] = df[col]
        
        # Align features
        for feature in target_features:
            if feature in available_features:
                # Feature exists, copy it
                aligned_df[feature] = df[feature]
            else:
                # Feature missing, fill with default value
                if fill_missing == "zero":
                    aligned_df[feature] = 0.0
                elif fill_missing == "mean":
                    # Use mean from available features (fallback to zero)
                    aligned_df[feature] = 0.0
                elif fill_missing == "median":
                    aligned_df[feature] = 0.0
                else:
                    aligned_df[feature] = 0.0
                
                logger.debug(f"Missing feature {feature}, filled with {fill_missing}")
        
        # Ensure all target features are present
        missing_targets = set(target_features) - set(aligned_df.columns)
        for feature in missing_targets:
            aligned_df[feature] = 0.0
        
        # Select only target features + metadata
        final_cols = target_features + [col for col in metadata_cols if col in aligned_df.columns]
        aligned_df = aligned_df[final_cols]
        
        logger.info(f"Aligned dataset: {len(df)} samples, {len(target_features)} features")
        
        return aligned_df
    
    def create_feature_mapping(
        self,
        source_features: List[str],
        target_features: List[str]
    ) -> Dict[str, str]:
        """
        Create a mapping from source features to target features
        
        Args:
            source_features: Source feature names
            target_features: Target feature names
            
        Returns:
            Dictionary mapping source -> target
        """
        mapping = {}
        
        # Exact matches
        for feat in source_features:
            if feat in target_features:
                mapping[feat] = feat
        
        # TODO: Add fuzzy matching for similar feature names
        # (e.g., "Flow Duration" vs "flow_duration" vs "FlowDuration")
        
        return mapping
    
    def compute_feature_stats(self, df: pd.DataFrame, features: List[str]) -> Dict:
        """Compute statistics for features"""
        stats = {}
        
        for feature in features:
            if feature in df.columns:
                stats[feature] = {
                    "mean": float(df[feature].mean()),
                    "std": float(df[feature].std()),
                    "min": float(df[feature].min()),
                    "max": float(df[feature].max()),
                    "median": float(df[feature].median()),
                    "missing": int(df[feature].isnull().sum()),
                    "missing_pct": float(df[feature].isnull().sum() / len(df) * 100)
                }
            else:
                stats[feature] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                    "missing": len(df),
                    "missing_pct": 100.0
                }
        
        return stats
    
    def save_feature_mapping(self, path: Path):
        """Save feature mapping to disk"""
        data = {
            "common_features": self.common_features,
            "feature_mapping": self.feature_mapping,
            "feature_stats": self.feature_stats
        }
        joblib.dump(data, path)
        logger.info(f"Feature mapping saved to {path}")
    
    def load_feature_mapping(self, path: Path):
        """Load feature mapping from disk"""
        data = joblib.load(path)
        self.common_features = data["common_features"]
        self.feature_mapping = data["feature_mapping"]
        self.feature_stats = data["feature_stats"]
        logger.info(f"Feature mapping loaded from {path}")


def align_multiple_datasets(
    datasets: List[pd.DataFrame],
    fill_missing: str = "zero"
) -> pd.DataFrame:
    """
    Align multiple datasets to common feature set
    
    Args:
        datasets: List of dataframes from different datasets
        fill_missing: How to fill missing features
        
    Returns:
        Combined aligned dataframe
    """
    aligner = FeatureAligner()
    
    # Find common features
    common_features = aligner.find_common_features(datasets)
    
    if not common_features:
        raise ValueError("No common features found across datasets")
    
    # Align each dataset
    aligned_datasets = []
    for i, df in enumerate(datasets):
        aligned_df = aligner.align_dataset(df, common_features, fill_missing=fill_missing)
        aligned_datasets.append(aligned_df)
        logger.info(f"Aligned dataset {i}: {len(aligned_df)} samples, {len(common_features)} features")
    
    # Combine aligned datasets
    combined_df = pd.concat(aligned_datasets, ignore_index=True)
    logger.info(f"Combined aligned dataset: {len(combined_df)} samples, {len(common_features)} features")
    
    return combined_df, common_features

