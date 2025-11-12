"""
Enhanced Data Loader for Multiple Cybersecurity Datasets

Supports:
- CICIDS2017 (existing)
- CIC IoT-IDAD Dataset 2024
- CICAPT-IIOT
- Global Cybersecurity Threats (2015-2024)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    path: Path
    label_column: str
    file_pattern: str = "*.csv"
    encoding: str = "utf-8"
    separator: str = ","
    sample_size: Optional[int] = None


class MultiDatasetLoader:
    """Loader for multiple cybersecurity datasets"""
    
    def __init__(self, base_data_dir: Optional[Path] = None):
        """
        Initialize multi-dataset loader
        
        Args:
            base_data_dir: Base directory containing all datasets
        """
        if base_data_dir is None:
            self.project_root = Path(__file__).parent.parent
            self.base_data_dir = self.project_root / "data"
        else:
            self.base_data_dir = Path(base_data_dir)
        
        # Dataset configurations
        self.datasets = {
            "cicids2017": DatasetInfo(
                name="CICIDS2017",
                path=self.base_data_dir / "cicids2017" / "MachineLearningCCSV",
                label_column="Label",
                file_pattern="*.csv"
            ),
            "cic_iot_2024": DatasetInfo(
                name="CIC IoT-IDAD 2024",
                path=self.base_data_dir / "cic_iot_idad_2024" / "Dataset",
                label_column="Label",  # May need adjustment based on actual dataset
                file_pattern="*.csv"
            ),
            "cicapt_iiot": DatasetInfo(
                name="CICAPT-IIOT",
                path=self.base_data_dir / "cicapt_iiot",
                label_column="Label",  # May need adjustment
                file_pattern="*.csv"
            ),
            "global_threats": DatasetInfo(
                name="Global Cybersecurity Threats",
                path=self.base_data_dir / "global_cybersecurity_threats",
                label_column="Label",  # May need adjustment
                file_pattern="*.csv"
            )
        }
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets"""
        available = []
        for key, info in self.datasets.items():
            if info.path.exists():
                available.append(key)
            else:
                logger.warning(f"Dataset {info.name} not found at {info.path}")
        return available
    
    def explore_dataset(self, dataset_key: str) -> Dict:
        """
        Explore a dataset to understand its structure
        
        Args:
            dataset_key: Key of the dataset to explore
            
        Returns:
            Dictionary with dataset information
        """
        if dataset_key not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        info = self.datasets[dataset_key]
        result = {
            "name": info.name,
            "path": str(info.path),
            "exists": info.path.exists(),
            "files": [],
            "columns": None,
            "sample_rows": 0,
            "label_values": None
        }
        
        if not info.path.exists():
            return result
        
        # Find all CSV files
        csv_files = list(info.path.rglob(info.file_pattern))
        result["files"] = [str(f) for f in csv_files]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {info.path}")
            return result
        
        # Load a sample file to understand structure
        try:
            sample_file = csv_files[0]
            logger.info(f"Exploring sample file: {sample_file}")
            
            # Try to load with different encodings
            df_sample = None
            for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
                try:
                    df_sample = pd.read_csv(
                        sample_file,
                        nrows=1000,
                        encoding=encoding,
                        sep=info.separator,
                        low_memory=False
                    )
                    result["encoding"] = encoding
                    break
                except Exception as e:
                    logger.debug(f"Failed with encoding {encoding}: {e}")
                    continue
            
            if df_sample is None:
                logger.error(f"Could not load sample file: {sample_file}")
                return result
            
            result["columns"] = list(df_sample.columns)
            result["sample_rows"] = len(df_sample)
            result["dtypes"] = {str(k): str(v) for k, v in df_sample.dtypes.to_dict().items()}
            
            # Check for label column
            label_col = self._find_label_column(df_sample, info.label_column)
            if label_col:
                result["label_column"] = label_col
                result["label_values"] = df_sample[label_col].value_counts().to_dict()
                result["label_distribution"] = df_sample[label_col].value_counts(normalize=True).to_dict()
            
            # Get numeric columns
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
            result["numeric_columns"] = numeric_cols
            result["num_features"] = len(numeric_cols)
            
            # Check for missing values
            result["missing_values"] = df_sample.isnull().sum().to_dict()
            result["missing_percentage"] = (df_sample.isnull().sum() / len(df_sample) * 100).to_dict()
            
        except Exception as e:
            logger.error(f"Error exploring dataset: {e}", exc_info=True)
            result["error"] = str(e)
        
        return result
    
    def _find_label_column(self, df: pd.DataFrame, default_label: str) -> Optional[str]:
        """Find the label column in the dataframe"""
        # Try default label
        if default_label in df.columns:
            return default_label
        
        # Try common label column names
        label_candidates = ["Label", "label", " Label", "LABEL", "target", "Target", "class", "Class"]
        for candidate in label_candidates:
            if candidate in df.columns:
                return candidate
        
        # Try columns with "label" in name (case-insensitive)
        for col in df.columns:
            if "label" in str(col).lower():
                return col
        
        return None
    
    def _process_labels(self, df: pd.DataFrame, dataset_key: str, label_column: str, file_path: Path) -> pd.DataFrame:
        """
        Process and standardize labels based on dataset type
        
        Args:
            df: Dataframe with labels
            dataset_key: Key of the dataset
            label_column: Name of the label column
            file_path: Path to the file (for folder-based label inference)
            
        Returns:
            Dataframe with processed labels
        """
        if label_column not in df.columns:
            return df
        
        # CIC IoT-IDAD 2024: Infer labels from folder structure
        if dataset_key == "cic_iot_2024":
            # Check if labels need to be inferred from folder structure
            if df[label_column].dtype == 'object':
                # Check for "NeedManualLabel" or similar
                unique_labels = df[label_column].unique()
                if len(unique_labels) == 1 and "NeedManualLabel" in str(unique_labels[0]):
                    # Infer label from folder structure
                    folder_name = file_path.parent.name
                    
                    # Map folder names to labels
                    folder_to_label = {
                        "Benign": "BENIGN",
                        "DOS": "DOS",
                        "DDOS": "DDOS",
                        "Mirai": "Mirai",
                        "Brute Force": "BruteForce",
                        "Recon": "Recon",
                        "Spoofing": "Spoofing"
                    }
                    
                    # Get label from folder name
                    inferred_label = folder_to_label.get(folder_name, "ATTACK")
                    df[label_column] = inferred_label
                    logger.info(f"Inferred label '{inferred_label}' from folder '{folder_name}' for file {file_path.name}")
            
            # Convert all non-BENIGN to ATTACK for binary classification
            df[label_column] = df[label_column].apply(
                lambda x: "BENIGN" if str(x).upper() == "BENIGN" else "ATTACK"
            )
        
        # CICAPT-IIOT: Convert numeric labels to text
        elif dataset_key == "cicapt_iiot":
            if df[label_column].dtype in ['int64', 'float64']:
                # Convert: 0 → BENIGN, 1+ → ATTACK
                df[label_column] = df[label_column].apply(
                    lambda x: "BENIGN" if x == 0 else "ATTACK"
                )
                logger.info(f"Converted numeric labels to text for {file_path.name}")
        
        # CICIDS2017: Standardize existing labels
        elif dataset_key == "cicids2017":
            # Convert to binary: BENIGN vs ATTACK
            df[label_column] = df[label_column].apply(
                lambda x: "BENIGN" if str(x).strip().upper() == "BENIGN" else "ATTACK"
            )
        
        # Global Cybersecurity Threats: Skip (no labels)
        elif dataset_key == "global_threats":
            # This dataset doesn't have labels, skip processing
            logger.warning(f"Dataset {dataset_key} has no labels, skipping label processing")
        
        return df
    
    def load_dataset(
        self,
        dataset_key: str,
        sample_size: Optional[int] = None,
        file_limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load a specific dataset
        
        Args:
            dataset_key: Key of the dataset to load
            sample_size: Maximum number of samples to load (None = all)
            file_limit: Maximum number of files to load (None = all)
            
        Returns:
            Combined dataframe from all files in the dataset
        """
        if dataset_key not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(self.datasets.keys())}")
        
        info = self.datasets[dataset_key]
        
        if not info.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {info.path}")
        
        # Find all CSV files
        csv_files = list(info.path.rglob(info.file_pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {info.path}")
        
        if file_limit:
            csv_files = csv_files[:file_limit]
        
        logger.info(f"Loading {len(csv_files)} files from {info.name}")
        
        dfs = []
        total_rows = 0
        
        for file_path in csv_files:
            try:
                logger.info(f"Loading file: {file_path.name}")
                
                # Try different encodings
                df = None
                for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
                    try:
                        if sample_size and total_rows >= sample_size:
                            break
                        
                        rows_to_read = None
                        if sample_size:
                            remaining = sample_size - total_rows
                            if remaining > 0:
                                rows_to_read = min(remaining, 100000)  # Read in chunks
                        
                        df = pd.read_csv(
                            file_path,
                            nrows=rows_to_read,
                            encoding=encoding,
                            sep=info.separator,
                            low_memory=False
                        )
                        break
                    except Exception as e:
                        logger.debug(f"Failed with encoding {encoding}: {e}")
                        continue
                
                if df is None:
                    logger.warning(f"Could not load file: {file_path}")
                    continue
                
                # Find and standardize label column
                label_col = self._find_label_column(df, info.label_column)
                if label_col and label_col != info.label_column:
                    df = df.rename(columns={label_col: info.label_column})
                
                # Handle label conversion based on dataset
                df = self._process_labels(df, dataset_key, info.label_column, file_path)
                
                # Add dataset source column
                df["_dataset_source"] = dataset_key
                
                dfs.append(df)
                total_rows += len(df)
                
                if sample_size and total_rows >= sample_size:
                    break
                    
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"Could not load any data from {info.name}")
        
        # Combine all dataframes
        df_combined = pd.concat(dfs, ignore_index=True)
        
        if sample_size and len(df_combined) > sample_size:
            df_combined = df_combined.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df_combined)} samples from {info.name}")
        
        # Show label distribution if label column exists
        if info.label_column in df_combined.columns:
            label_dist = df_combined[info.label_column].value_counts()
            logger.info(f"Label distribution:\n{label_dist}")
            
            # Check if we have both BENIGN and ATTACK labels
            label_values = [str(v).upper() for v in label_dist.index]
            has_benign = any("BENIGN" in v for v in label_values)
            has_attack = any("ATTACK" in v for v in label_values)
            
            if not has_benign or not has_attack:
                logger.warning(f"Dataset {dataset_key} may not have both BENIGN and ATTACK labels")
                logger.warning(f"Found labels: {list(label_dist.index)[:10]}")
        
        return df_combined
    
    def load_multiple_datasets(
        self,
        dataset_keys: List[str],
        sample_size_per_dataset: Optional[int] = None,
        file_limit_per_dataset: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load and combine multiple datasets
        
        Args:
            dataset_keys: List of dataset keys to load
            sample_size_per_dataset: Maximum samples per dataset
            file_limit_per_dataset: Maximum files per dataset
            
        Returns:
            Combined dataframe from all datasets
        """
        dfs = []
        
        for key in dataset_keys:
            try:
                df = self.load_dataset(
                    key,
                    sample_size=sample_size_per_dataset,
                    file_limit=file_limit_per_dataset
                )
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading dataset {key}: {e}")
                continue
        
        if not dfs:
            raise ValueError("Could not load any datasets")
        
        # Combine all datasets
        df_combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(df_combined)} total samples from {len(dataset_keys)} datasets")
        
        return df_combined
    
    def get_dataset_info(self, dataset_key: str) -> DatasetInfo:
        """Get dataset information"""
        if dataset_key not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        return self.datasets[dataset_key]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    loader = MultiDatasetLoader()
    
    # List available datasets
    print("Available datasets:")
    available = loader.list_available_datasets()
    for key in available:
        print(f"  - {key}")
    
    # Explore a dataset
    if available:
        print(f"\nExploring dataset: {available[0]}")
        info = loader.explore_dataset(available[0])
        print(f"Columns: {len(info.get('columns', []))}")
        print(f"Features: {info.get('num_features', 0)}")
        print(f"Label column: {info.get('label_column', 'Not found')}")
        if info.get('label_values'):
            print(f"Label distribution: {info['label_values']}")

