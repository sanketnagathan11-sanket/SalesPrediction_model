"""
End-to-End Execution Pipeline for Sales Prediction.
CodeAlpha Data Science Internship - Task 4

Usage:
    python main.py
"""

import sys
import os

# Ensure package root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.data_loader import SalesDataLoader
from src.visualize import SalesVisualizer
from src.model import SalesModelPipeline
from src.predict import predict_sales


def run_pipeline():
    print("\n" + "#" * 65)
    print("    CODEALPHA DATA SCIENCE INTERNSHIP: TASK 4 (SALES PREDICTION)    ")
    print("#" * 65 + "\n")

    # Step 1: Data Loading & Exploration
    print("[1/5] Ingesting and inspecting advertising dataset...")
    loader = SalesDataLoader()
    df = loader.load_data()
    print("First 5 records:")
    print(df.head().to_string())
    print("\nStatistical Summary:")
    print(df.describe().round(2).to_string())

    # Step 2: Visualization & EDA
    print("\n[2/5] Generating exploratory advertising regression plots...")
    visualizer = SalesVisualizer()
    visualizer.plot_regression_trends(df)
    visualizer.plot_correlation_heatmap(df)

    # Step 3: Train / Test Split
    print("\n[3/5] Partitioning data into train and test sets...")
    X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.2, random_state=42)

    # Step 4: Model Training, Benchmarking & Evaluation
    print("\n[4/5] Training regression models and benchmarking...")
    pipeline = SalesModelPipeline()
    results = pipeline.train_and_evaluate_all(X_train, X_test, y_train, y_test)

    # Visual diagnostics for best model
    best_name = pipeline.best_model_name
    best_pred = results[best_name]["y_pred_test"]
    visualizer.plot_actual_vs_predicted(y_test, best_pred, model_name=best_name)
    visualizer.plot_residuals(y_test, best_pred)

    # Save model artifact
    model_artifact_path = pipeline.save_best_model(X_train.columns)

    # Step 5: Verification & Sample Inferences
    print("\n[5/5] Running strategic budget scenario forecasts...")
    scenarios = [
        {"desc": "Balanced Portfolio", "tv": 150.0, "radio": 25.0, "newspaper": 15.0},
        {"desc": "Heavy TV Focus", "tv": 250.0, "radio": 10.0, "newspaper": 5.0},
        {"desc": "Digital/Radio Heavy", "tv": 50.0, "radio": 45.0, "newspaper": 10.0}
    ]

    for sc in scenarios:
        sales, m_name = predict_sales(sc["tv"], sc["radio"], sc["newspaper"], model_path=model_artifact_path)
        print(f"Scenario [{sc['desc']:<20}]: TV=${sc['tv']}k, Radio=${sc['radio']}k, Newspaper=${sc['newspaper']}k => Forecast: {sales:.2f}k units ({m_name})")

    print("\n" + "#" * 65)
    print("    TASK 4 PIPELINE COMPLETED SUCCESSFULLY!    ")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
