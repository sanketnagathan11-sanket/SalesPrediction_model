"""
Model Training, Benchmark, and Evaluation for Sales Prediction.
CodeAlpha Data Science Internship - Task 4
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class SalesModelPipeline:
    """Manages training, multi-algorithm regression benchmarking, and serialization."""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.trained_models = {}
        self.evaluation_results = {}
        self.best_model_name = None

    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test):
        """Trains Linear Regression, Ridge, and Random Forest Regressors, comparing metrics."""
        candidate_models = {
            "Linear Regression (OLS)": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        best_r2 = -float("inf")

        print("=" * 65)
        print("            SALES PREDICTION REGRESSION BENCHMARK")
        print("=" * 65)

        for name, reg in candidate_models.items():
            reg.fit(X_train, y_train)
            self.trained_models[name] = reg

            y_pred_train = reg.predict(X_train)
            y_pred_test = reg.predict(X_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)

            self.evaluation_results[name] = {
                "train_r2": r2_train,
                "test_r2": r2_test,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "y_pred_test": y_pred_test
            }

            print(f"\n--- {name} ---")
            print(f"Train R² Score : {r2_train:.4f}")
            print(f"Test R² Score  : {r2_test:.4f}")
            print(f"MAE            : {mae:.4f}")
            print(f"MSE            : {mse:.4f}")
            print(f"RMSE           : {rmse:.4f}")

            if r2_test > best_r2:
                best_r2 = r2_test
                self.best_model_name = name

        print("=" * 65)
        print(f"Best Performing Model: {self.best_model_name} with Test R²: {best_r2:.4f}")
        print("=" * 65)

        # Print marketing coefficient insights from Linear Regression
        lr = self.trained_models["Linear Regression (OLS)"]
        print("\nMarketing Budget Channel Impact (Linear Regression Coefficients):")
        print(f"  Intercept (Base Sales without ads) : {lr.intercept_:.4f}")
        for feat, coef in zip(X_train.columns, lr.coef_):
            print(f"  - Each $1,000 in {feat:<9} spend yields {coef:+.4f}k sales units")

        return self.evaluation_results

    def save_best_model(self, feature_names):
        """Serializes the best model into a joblib bundle."""
        if not self.best_model_name:
            raise ValueError("No model has been trained yet!")

        artifact = {
            "model_name": self.best_model_name,
            "model": self.trained_models[self.best_model_name],
            "feature_names": list(feature_names),
            "linear_model": self.trained_models["Linear Regression (OLS)"]
        }
        save_path = os.path.join(self.models_dir, "sales_best_model.joblib")
        joblib.dump(artifact, save_path)
        print(f"\nSerialized best model bundle saved to: {save_path}")
        return save_path
