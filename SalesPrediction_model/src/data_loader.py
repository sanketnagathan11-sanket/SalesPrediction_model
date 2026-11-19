"""
Data Ingestion and Preprocessing Module for Sales Prediction.
CodeAlpha Data Science Internship - Task 4
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


class SalesDataLoader:
    """Handles ingestion, sanitization, and splitting of advertising sales data."""

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "advertising.csv")
        self.data_path = data_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Loads advertising sales data, drops unnecessary unnamed index columns, and validates values."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Advertising dataset not found at: {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # Drop unnamed index column if present
        unnamed_cols = [c for c in self.df.columns if "Unnamed" in c or c == ""]
        if unnamed_cols:
            self.df = self.df.drop(columns=unnamed_cols)

        # Ensure correct column names
        expected_cols = ["TV", "Radio", "Newspaper", "Sales"]
        for col in expected_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing expected column '{col}' in dataset.")

        # Ensure numeric datatypes
        self.df = self.df[expected_cols].apply(pd.to_numeric, errors="coerce")

        # Null check
        null_count = self.df.isnull().sum().sum()
        if null_count > 0:
            print(f"Warning: Found {null_count} nulls. Dropping missing entries...")
            self.df = self.df.dropna()

        print(f"Dataset successfully loaded: {self.df.shape[0]} records, {self.df.shape[1]} columns.")
        return self.df

    def get_features_and_target(self):
        """Extracts predictor matrix X and response vector y."""
        if self.df is None:
            self.load_data()

        X = self.df[["TV", "Radio", "Newspaper"]]
        y = self.df["Sales"]
        return X, y

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Partitions data into training and test sets."""
        X, y = self.get_features_and_target()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train/Test Partition: Train ({len(X_train)} samples), Test ({len(X_test)} samples).")
        return X_train, X_test, y_train, y_test
