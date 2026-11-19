# CodeAlpha - Sales Prediction using Python (Task 4)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

---

## 📌 Project Overview
This repository contains the complete implementation for **Task 4: Sales Prediction using Python** as part of the **CodeAlpha Data Science Internship**.

The objective of this project is to model and forecast future sales based on advertising investments across three primary media platforms:
1. **TV Advertising Spend ($'000)**
2. **Radio Advertising Spend ($'000)**
3. **Newspaper Advertising Spend ($'000)**

We quantify channel effectiveness, identify return on investment (ROI), and deliver an interactive forecasting pipeline to support data-driven marketing decisions.

---

## 📂 Project Architecture

```
CodeAlpha_SalesPrediction/
│
├── data/
│   └── advertising.csv                     # Raw advertising expenditure dataset (200 records)
├── models/
│   └── sales_best_model.joblib             # Serialized model & metadata bundle
├── notebooks/
│   └── sales_prediction.ipynb             # Interactive analysis & modeling notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                      # Data loading, cleaning & train/test partitioning
│   ├── model.py                            # OLS, Ridge & Random Forest benchmarking
│   ├── predict.py                          # Interactive CLI sales forecast tool
│   └── visualize.py                        # Regression trendlines, heatmap & diagnostic plots
├── visuals/
│   ├── advertising_channels_vs_sales.png   # Scatter + regression plots per channel
│   ├── correlation_heatmap.png             # Correlation matrix across channels & sales
│   ├── actual_vs_predicted_sales.png       # Actual vs. predicted fit comparison
│   └── residuals_distribution.png          # Residual errors and normality check
├── main.py                                 # End-to-end execution pipeline
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/CodeAlpha_SalesPrediction.git
   cd CodeAlpha_SalesPrediction
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Run the Full Automated Pipeline
To execute data ingestion, exploratory regression visualization, model benchmarking, diagnostic plotting, and scenario forecasting:
```bash
python main.py
```

### 2. Interactive Sales Forecasting Tool
Run the inference script to forecast sales for any marketing budget:
```bash
python src/predict.py --tv 150.0 --radio 25.0 --newspaper 15.0
```
Example Output:
```
==================================================
           SALES FORECAST RESULT
==================================================
Model Applied   : Random Forest Regressor
Marketing Budget: TV=$150.0k | Radio=$25.0k | Newspaper=$15.0k
Total Ad Budget : $190.0k
Forecasted Sales: >>> 15.19k units <<< (Approx 15,190 units)
==================================================
```

---

## 📊 Exploratory Data Analysis & Strategic Insights
- **TV Dominance**: TV spend has the strongest overall linear correlation with sales ($r = 0.782$).
- **Radio High Multiplier**: Each $1,000 spent on Radio advertising yields **+0.1892k sales units**, demonstrating the highest per-dollar return rate.
- **Newspaper Inefficiency**: Newspaper advertising shows negligible direct influence ($+0.0028k$ sales per $1,000$), suggesting budgets should be reallocated towards TV and Radio for maximum ROI.

---

## 📈 Model Comparison & Benchmark Results

| Model | Train $R^2$ | Test $R^2$ | Test MAE | Test MSE | Test RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | 0.8957 | **0.8994** | 1.4608 | 3.1741 | 1.7816 |
| **Ridge Regression ($\alpha=1.0$)** | 0.8957 | **0.8994** | 1.4608 | 3.1741 | 1.7816 |
| **Random Forest Regressor** | 0.9964 | **0.9813** | **0.6203** | **0.5909** | **0.7687** |

The **Random Forest Regressor** achieved superior non-linear capture with a **$98.13\%$ explanatory variance ($R^2 = 0.9813$)** and low RMSE of **0.7687**.

---

## 👨‍💻 Author & Acknowledgements
- **Author**: Shreya (CodeAlpha Data Science Intern)
- **Organization**: [CodeAlpha](https://www.codealpha.tech)
