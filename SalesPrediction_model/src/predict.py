"""
Interactive CLI Prediction Module for Sales Forecasting.
CodeAlpha Data Science Internship - Task 4
"""

import os
import argparse
import joblib
import pandas as pd


def load_sales_artifact(model_path: str = None):
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "sales_best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run main.py first.")
    return joblib.load(model_path)


def predict_sales(tv: float, radio: float, newspaper: float, model_path: str = None):
    """Predicts expected sales in thousands of units given media budgets."""
    artifact = load_sales_artifact(model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    input_df = pd.DataFrame([[tv, radio, newspaper]], columns=feature_names)
    predicted_sales = model.predict(input_df)[0]
    return predicted_sales, artifact["model_name"]


def main():
    parser = argparse.ArgumentParser(description="Forecast product sales from marketing spend.")
    parser.add_argument("--tv", type=float, default=None, help="TV Advertising Budget ($'000)")
    parser.add_argument("--radio", type=float, default=None, help="Radio Advertising Budget ($'000)")
    parser.add_argument("--newspaper", type=float, default=None, help="Newspaper Advertising Budget ($'000)")

    args = parser.parse_args()

    if None in (args.tv, args.radio, args.newspaper):
        print("\n--- Sales Prediction & Ad Impact Forecaster ---")
        print("Enter advertising budgets in thousands of dollars ($'000):")
        tv = float(input("TV Spend ($'000) [e.g. 150.0]: ") or 150.0)
        radio = float(input("Radio Spend ($'000) [e.g. 25.0]: ") or 25.0)
        newspaper = float(input("Newspaper Spend ($'000) [e.g. 15.0]: ") or 15.0)
    else:
        tv, radio, newspaper = args.tv, args.radio, args.newspaper

    sales, model_name = predict_sales(tv, radio, newspaper)

    print("\n" + "=" * 50)
    print("           SALES FORECAST RESULT")
    print("=" * 50)
    print(f"Model Applied   : {model_name}")
    print(f"Marketing Budget: TV=${tv:,.1f}k | Radio=${radio:,.1f}k | Newspaper=${newspaper:,.1f}k")
    print(f"Total Ad Budget : ${tv + radio + newspaper:,.1f}k")
    print(f"Forecasted Sales: >>> {sales:.2f}k units <<< (Approx {int(sales * 1000):,} units)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
