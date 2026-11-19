"""
Visualization Module for Sales Prediction.
Produces regression trendlines, correlation matrices, and model diagnostic plots.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SalesVisualizer:
    """Generates and saves exploratory and predictive charts for Sales forecasting."""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "visuals")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="deep")

    def plot_regression_trends(self, df):
        """Plots scatter plots with OLS regression lines for TV, Radio, and Newspaper vs. Sales."""
        save_path = os.path.join(self.output_dir, "advertising_channels_vs_sales.png")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

        channels = ["TV", "Radio", "Newspaper"]
        colors = ["#2b5c8f", "#d95f02", "#7570b3"]

        for ax, ch, col in zip(axes, channels, colors):
            sns.regplot(data=df, x=ch, y="Sales", ax=ax, color=col,
                        scatter_kws={"alpha": 0.6}, line_kws={"color": "red", "linewidth": 2})
            ax.set_title(f"{ch} Advertising vs Sales", fontsize=12, weight="bold")
            ax.set_xlabel(f"{ch} Budget ($'000)", fontsize=11)
            ax.set_ylabel("Sales Units ('000)", fontsize=11)

        plt.suptitle("Impact of Media Advertising Channels on Sales Outcomes", fontsize=14, weight="bold", y=1.03)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_correlation_heatmap(self, df):
        """Plots correlation matrix heatmap of ad channels and sales."""
        save_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.figure(figsize=(7, 5))
        corr = df[["TV", "Radio", "Newspaper", "Sales"]].corr()
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.8, vmin=-1, vmax=1)
        plt.title("Correlation Matrix: Media Spend vs Sales", fontsize=12, weight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_actual_vs_predicted(self, y_actual, y_predicted, model_name: str = "Linear Regression"):
        """Plots actual vs predicted sales with perfect prediction reference line."""
        save_path = os.path.join(self.output_dir, "actual_vs_predicted_sales.png")
        plt.figure(figsize=(7, 6))

        plt.scatter(y_actual, y_predicted, color="#1f77b4", alpha=0.7, edgecolors="k", s=60, label="Predictions")
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y = x)")

        plt.title(f"Actual vs. Predicted Sales ({model_name})", fontsize=12, weight="bold")
        plt.xlabel("Actual Sales ('000 units)", fontsize=11, weight="bold")
        plt.ylabel("Predicted Sales ('000 units)", fontsize=11, weight="bold")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_residuals(self, y_actual, y_predicted):
        """Plots residual distribution to evaluate regression error normality."""
        save_path = os.path.join(self.output_dir, "residuals_distribution.png")
        residuals = y_actual - y_predicted

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals vs Predicted
        ax1.scatter(y_predicted, residuals, color="#e6550d", alpha=0.7, edgecolors="k")
        ax1.axhline(0, color="black", linestyle="--", linewidth=1.5)
        ax1.set_title("Residuals vs. Fitted Values", fontsize=11, weight="bold")
        ax1.set_xlabel("Fitted Values", fontsize=10)
        ax1.set_ylabel("Residuals", fontsize=10)

        # Residuals histogram
        sns.histplot(residuals, kde=True, ax=ax2, color="#3182bd")
        ax2.set_title("Distribution of Residuals", fontsize=11, weight="bold")
        ax2.set_xlabel("Residual Error", fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
