# Alpha Allocator: Portfolio Optimizer

### Research Question
*Can a ML model, trained on sector-relative technical indicators, identify stocks that statistically outperform a passive market-cap-weighted benchmark?*

---

## Overview
The **Alpha Allocator** is a machine learning pipeline that attempts to predict raw stock returns using a **Sector-Relative** approach.

It trains a **Histogram Gradient Boosting Classifier** to identify stocks that are likely to outperform their specific sector over a 3-month horizon. The system outputs a high-conviction "Buy" list which feeds into a dynamic portfolio simulation engine, applying aggressive capital allocation to high-conviction assets while comparing with a passive investement in the S&P500.

## Key Features
* **Advanced ML Model:** Uses Histogram Gradient Boosting with `RandomizedSearchCV` for parameter tuning.
* **Sector-Relative Targeting:** "Curves the grades" by comparing stocks only against their sector peers, effectively neutralizing market beta and regime bias.
* **Monte Carlo Simulation:** An engine running 200 simulation paths to project portfolio performance, VaR (Value at Risk), and Sharpe Ratio over a 1-year horizon.
* **Interactive Dashboard:** A streamlit application allowing users to visualize tactical picks, adjust risk profiles, and set conviction thresholds.

## Project Pipeline
The pipeline is orchestrated by `main.py` and executes the full end-to-end process:

1.  **Data:** Fetches 10 years of daily adjusted closing prices for ~320 US equities and Macro ETFs via `yfinance` (`src/data_processing.py`).
2.  **Features:** Computes "Smart Beta" indicators (RSI, Momentum, Trend Distance) and generates sector-relative targets.
3.  **Model:** Optimizes a Histogram Gradient Boosting Classifier using Time-Series Cross-Validation (`src/optimize_model.py`).
4.  **Audit:** Evaluates the model on out-of-sample data (2024-2025) and generates confusion matrices (`src/visualize_performance.py`).
5.  **Simulation & Dashboard:** Runs Monte Carlo simulations and launches the interactive user interface (`src/app.py` & `src/portfolio_simulation.py`).

## Project Structure
```text
├── main.py                 # ENTRY POINT: Orchestrates the entire pipeline
├── requirements.txt        # Pip dependency list
├── environment_full.yml    # Conda environment specification
├── README.md               # Project documentation
├── src/                    # Source code
│   ├── data_processing.py      # Fetches YFinance data & calculates features
│   ├── optimize_model.py       # Trains Gradient Boosting model & tunes parameters
│   ├── portfolio_simulation.py # Runs Monte Carlo simulations & asset allocation logic
│   ├── visualize_performance.py# Generates audit charts (Confusion Matrix)
│   └── app.py                  # Streamlit Dashboard code
├── data/                   # (Auto-Generated) Local storage for downloaded stock data
└── results/                # (Auto-Generated) Stores performance plots and audits
```

## How to Run
1. Clone the repository
    ```text
    git clone https://github.com/PladimirVutin1991/alpha-allocator.git
    cd alpha-allocator
    ```
2. Create the Environment

    This project relies on specific versions of scientific computing libraries. Create the environment using Conda:
    ```text
    conda env create -f environment_full.yml
    conda activate alpha-allocator
    ```
    Note: Use environment_full.yml NOT environment.yml. If it does not work try 
    ```text
    pip install -r requirements.txt
    ```

3. Run the Pipeline
    ```text
    python main.py
    ```
    Note: At the end if you run it the first time streamlit will ask you to register with an email. Just press Enter. The dashboard will then pop up either automatically or through a popup on the down-right corner.

## Requirements
- Python 3.9+
- pandas
- numpy
- scikit-learn
- yfinance
- plotly
- streamlit
- joblib
- matplotlib
- seaborn
