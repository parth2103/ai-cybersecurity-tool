import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.project_root = Path(__file__).parent.parent
        
    def clean_data(self, df):
        """Clean the dataset"""
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle infinity and NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_len - len(df)} duplicates")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and labels"""
        # Get label column (handle both 'Label' and ' Label')
        label_col = ' Label' if ' Label' in df.columns else 'Label'
        
        # Binary classification
        y = (df[label_col] != 'BENIGN').astype(int)
        
        # Features
        X = df.drop([label_col], axis=1)
        X = X.select_dtypes(include=[np.number])
        
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Attack ratio: {y.mean():.2%}")
        
        return X, y
    
    def process_data(self, df, test_size=0.2):
        """Complete preprocessing pipeline"""
        # Clean
        df = self.clean_data(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Save preprocessors
        models_dir = self.project_root / 'models'
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(self.scaler, models_dir / 'scaler.pkl')
        joblib.dump(list(X.columns), models_dir / 'feature_names.pkl')
        
        # Save processed data
        processed_dir = self.project_root / 'data' / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        np.save(processed_dir / 'X_train.npy', X_train)
        np.save(processed_dir / 'X_test.npy', X_test)
        np.save(processed_dir / 'y_train.npy', y_train)
        np.save(processed_dir / 'y_test.npy', y_test)
        
        print(f"\nProcessed data saved to {processed_dir}")
        
        return X_train, X_test, y_train, y_test