import pandas as pd
import numpy as np
import os
from pathlib import Path


class CICIDSDataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Use path relative to project root
            self.project_root = Path(__file__).parent.parent
            self.data_dir = (
                self.project_root / "data" / "cicids2017" / "MachineLearningCCSV"
            )
        else:
            self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            print(
                f"Warning: Data directory not found: {self.data_dir}. A synthetic dataset will be generated."
            )

    def load_friday_data(self, sample_size=None):
        """Load Friday data with optional sampling"""
        print(f"Loading data from: {self.data_dir}")

        # File paths
        ddos_file = self.data_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        portscan_file = (
            self.data_dir / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        )

        # Load with optional sampling
        try:
            if sample_size:
                df_ddos = pd.read_csv(ddos_file, nrows=sample_size // 2)
                df_portscan = pd.read_csv(portscan_file, nrows=sample_size // 2)
            else:
                df_ddos = pd.read_csv(ddos_file)
                df_portscan = pd.read_csv(portscan_file)
            df = pd.concat([df_ddos, df_portscan], ignore_index=True)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Warning: {e}. Falling back to synthetic dataset for development.")
            df = self._generate_synthetic(sample_size or 10000)

        print(f"Loaded {len(df)} samples")
        # Column may be 'Label' or ' Label' depending on source; handle both
        label_col = (
            "Label"
            if "Label" in df.columns
            else (" Label" if " Label" in df.columns else None)
        )
        if label_col is not None:
            print(f"Attack distribution:\n{df[label_col].value_counts()}")

        return df

    def _generate_synthetic(self, num_rows: int) -> pd.DataFrame:
        """Generate a simple synthetic dataset resembling CICIDS structure."""
        rng = np.random.default_rng(42)
        num_features = 20
        X = rng.normal(0, 1, size=(num_rows, num_features))
        # Create a binary label with class imbalance
        attack_ratio = 0.3
        y = (rng.random(num_rows) < attack_ratio).astype(int)
        # Slightly shift means for attack samples to make classification feasible
        X[y == 1] += 0.8
        columns = [f"f{i}" for i in range(num_features)]
        df = pd.DataFrame(X, columns=columns)
        df["Label"] = np.where(y == 1, "ATTACK", "BENIGN")
        return df
