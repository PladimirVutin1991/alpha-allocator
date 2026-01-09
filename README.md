# Alpha Allocator: Portfolio Optimizer

### Research Question
*Can a ML model, trained on sector-relative technical indicators, identify stocks that statistically outperform a passive market-cap-weighted benchmark?*

---

## Project Overview
The **Alpha Allocator** is a ML pipeline that uses a **Sector-Relative** approach to try to predict raw stock returns (which is notoriously difficult). It trains a **Histogram Gradient Boosting Classifier** to identify stocks that are likely to outperform their specific sector peers over a 3-month horizon.

The system outputs a high-conviction "Buy" list which feeds into a portfolio simulation engine. The strategy applies aggressive capital allocation to these high-conviction assets while maintaining a core-satellite approach with passive ETFs for stability.

## Key Features
* **ML Model:** Uses Histogram Gradient Boosting with `RandomizedSearchCV` for hyperparameter tuning.
* **Sector-Relative Targeting:** "Curves the grades" by comparing stocks only against their sector peers, mitigating market regime bias.
* **Simulation:** A Monte Carlo engine (200 paths) that projects portfolio performance, VaR (Value at Risk), and Sharpe Ratio over a 1-year horizon.
* **Dashboard:** A full-stack Streamlit app allowing users to adjust risk profiles (Growth/Balanced/Income) and conviction thresholds in real-time.

## Project Structure
```text
├── main.py                 # ENTRY POINT: Orchestrates the entire pipeline
├── requirements.txt        # Dependency list
├── README.md               # Project documentation
├── src/
│   ├── data_processing.py      # Fetches YFinance data & engineers features (RSI, Momentum)
│   ├── optimize_model.py       # Trains Gradient Boosting model & tunes hyperparameters
│   ├── portfolio_simulation.py # Runs Monte Carlo simulations & asset allocation logic
│   ├── visualize_performance.py# Generates audit charts (Confusion Matrix)
│   └── app.py                  # Streamlit Dashboard code
├── data/                   # (Auto-Generated) Stores downloaded stock data
└── results/                # (Auto-Generated) Stores performance plots and audits
```
## How to run

### Setup Environment
Clone the repository and install dependencies:

**Terminal**
- git clone https://github.com/PladimirVutin1991/alpha-allocator.git
- cd alpha-allocator.git
- conda env create -f environment.yml
- conda activate alpha-allocator

### Run the pipeline
python main.py
