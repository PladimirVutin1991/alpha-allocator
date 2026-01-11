# Alpha Allocator: Portfolio Optimizer

### Research Question
*Can a Machine Learning model, trained on sector-relative technical indicators, identify a 'top decile' of stocks that statistically outperform a passive market-cap-weighted benchmark?*

---

## Overview
The **Alpha Allocator** is a machine learning pipeline designed to bridge the gap between Data Science and Wealth Management. Unlike traditional models that attempt to predict raw stock returns (which is notoriously difficult due to market noise), this project uses a **Sector-Relative** approach.

It trains a **Histogram Gradient Boosting Classifier** to identify stocks that are likely to outperform their specific sector peers (e.g., Tech vs. Tech) over a 3-month horizon. The system outputs a high-conviction "Buy" list which feeds into a dynamic portfolio simulation engine, applying aggressive capital allocation to high-conviction assets while maintaining a core-satellite approach with passive ETFs for stability.

## Key Features
* **Advanced ML Model:** Uses Histogram Gradient Boosting with `RandomizedSearchCV` for hyperparameter tuning.
* **Sector-Relative Targeting:** "Curves the grades" by comparing stocks only against their sector peers, effectively neutralizing market beta and regime bias.
* **Monte Carlo Simulation:** A robust engine running 200 simulation paths to project portfolio performance, VaR (Value at Risk), and Sharpe Ratio over a 1-year horizon.
* **Interactive Dashboard:** A full-stack **Streamlit** application allowing users to visualize tactical picks, adjust risk profiles (Growth/Balanced/Income), and set conviction thresholds in real-time.

## Project Pipeline
The pipeline is orchestrated by `main.py` and executes the full end-to-end process:

1.  **Data Ingestion:** Fetches 10 years of daily adjusted closing prices for ~320 US equities and Macro ETFs via `yfinance` (`src/data_processing.py`).
2.  **Feature Engineering:** Computes "Smart Beta" indicators (RSI, Momentum, Trend Distance) and generates sector-relative targets.
3.  **Model Training:** Optimizes a Histogram Gradient Boosting Classifier using Time-Series Cross-Validation (`src/optimize_model.py`).
4.  **Performance Audit:** Evaluates the model on out-of-sample data (2024-2025) and generates confusion matrices (`src/visualize_performance.py`).
5.  **Simulation & Dashboard:** Runs Monte Carlo simulations and launches the interactive user interface (`src/app.py`).

## Project Structure
```text
alpha-allocator/
├── main.py                 # ENTRY POINT: Orchestrates the entire pipeline
├── requirements.txt        # Pip dependency list
├── environment_full.yml    # Conda environment specification
├── README.md               # Project documentation
├── src/                    # Source code
│   ├── data_processing.py      # Data fetching & feature engineering
│   ├── optimize_model.py       # Model training & hyperparameter tuning
│   ├── portfolio_simulation.py # Monte Carlo simulation logic
│   ├── visualize_performance.py# Generates audit charts
│   └── app.py                  # Streamlit Dashboard interface
├── data/                   # (Auto-Generated) Local storage for datasets
└── results/                # (Auto-Generated) Stores performance plots

How to Run
1. Clone the repository

git clone [https://github.com/PladimirVutin1991/alpha-allocator.git](https://github.com/PladimirVutin1991/alpha-allocator.git)
cd alpha-allocator

2. Create the Environment

This project relies on specific versions of scientific computing libraries. Create the environment using Conda:

conda env create -f environment_full.yml
conda activate alpha-allocator

3. Run the Pipeline

Execute the main script. This will automatically check dependencies, download data, train the model, and launch the dashboard.
Bash

python main.py

Requirements

Python 3.9+
pandas
numpy
scikit-learn
yfinance
plotly
streamlit
joblib
matplotlib
seaborn