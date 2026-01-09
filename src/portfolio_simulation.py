import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

ALLOCATIONS = {
    'Income':    {'Equity': 0.30, 'Fixed_Income': 0.60, 'Cash': 0.05, 'Alts': 0.05},
    'Balanced':  {'Equity': 0.50, 'Fixed_Income': 0.40, 'Cash': 0.05, 'Alts': 0.05},
    'Growth':    {'Equity': 0.70, 'Fixed_Income': 0.20, 'Cash': 0.05, 'Alts': 0.05}
}

DATA_DIR = 'data'

def load_data():
    try:
        preds = pd.read_csv(f'{DATA_DIR}/model_predictions.csv')
        stock_prices = pd.read_csv(f'{DATA_DIR}/stock_prices.csv', index_col=0, parse_dates=True)
        macro_prices = pd.read_csv(f'{DATA_DIR}/macro_prices.csv', index_col=0, parse_dates=True)
        caps = pd.read_csv(f'{DATA_DIR}/market_caps.csv', index_col=0)
        if caps.shape[1] >= 1: caps = caps.iloc[:, 0]
        return preds, stock_prices, macro_prices, caps
    except FileNotFoundError as e:
        print(f"❌ Missing Data: {e}")
        return None, None, None, None

def calculate_weights(preds, caps, stock_tickers, threshold=0.53, active=True):
    valid_tickers = [t for t in stock_tickers if t in caps.index]
    relevant_caps = caps[valid_tickers]
    total_cap = relevant_caps.sum()
    if total_cap == 0:
        base_weights = pd.Series(1/len(valid_tickers), index=valid_tickers)
    else:
        base_weights = relevant_caps / total_cap

    if not active: return base_weights

    if 'Date' in preds.columns:
        latest_preds = preds.sort_values('Date').groupby('Ticker').last()
    else:
        latest_preds = preds.groupby('Ticker').last()
    
    multipliers = pd.Series(1.0, index=valid_tickers)
    for t in valid_tickers:
        if t in latest_preds.index:
            signal = latest_preds.loc[t, 'Predicted_Signal']
            conf = latest_preds.loc[t, 'Confidence_Score']
            
            # Overweighting ML strategy
            if signal == 1 and conf > threshold:
                multipliers[t] = 5.0  # Overweight
            else:
                multipliers[t] = 0.0  # Zero Tolerance for bad stocks
    
    final_weights = base_weights * multipliers
    if final_weights.sum() == 0: return base_weights
    return final_weights / final_weights.sum()

def get_asset_vector(equity_dist, profile):
    policy = ALLOCATIONS[profile]
    equity_assets = list(equity_dist.index)
    bonds = ['AGG', 'LQD', 'HYG']
    alts = ['GLD', 'DBC', 'VNQ']
    cash = ['SHV']
    
    assets = equity_assets + bonds + alts + cash
    weights_vector = []
    
    for asset in assets:
        w = 0
        if asset in equity_assets:
            w = policy['Equity'] * equity_dist[asset]
        elif asset in bonds:
            w = policy['Fixed_Income'] / len(bonds)
        elif asset in alts:
            w = policy['Alts'] / len(alts)
        elif asset in cash:
            w = policy['Cash']
        weights_vector.append(w)
        
    return np.array(weights_vector), assets

def run_simulation(profile='Growth', threshold=0.53, is_dashboard=False):
    preds, stock_prices, macro_prices, caps = load_data()
    if preds is None: return None, None

    # ==========================================
    # --- 10 YEAR SIMULATION ---
    # ==========================================
    
    sim_start_date = '2015-01-01'
    
    full_prices = pd.concat([stock_prices, macro_prices], axis=1)
    full_prices = full_prices[full_prices.index >= sim_start_date]
    
    valid_cols = full_prices.columns[full_prices.notna().mean() > 0.9]
    full_prices = full_prices[valid_cols]
    full_prices = full_prices.ffill().dropna()
    
    returns = full_prices.pct_change().dropna()
    if returns.empty: return None, None

    surviving_stocks = [t for t in stock_prices.columns if t in valid_cols]
    
    eq_active = calculate_weights(preds, caps, surviving_stocks, threshold, active=True)
    w_active_raw, assets_raw = get_asset_vector(eq_active, profile)
    
    eq_passive = calculate_weights(preds, caps, surviving_stocks, threshold, active=False)
    w_passive_raw, _ = get_asset_vector(eq_passive, profile)

    final_assets = [a for a in assets_raw if a in returns.columns]
    
    w_active_final = []
    w_passive_final = []
    for asset in final_assets:
        idx = assets_raw.index(asset)
        w_active_final.append(w_active_raw[idx])
        w_passive_final.append(w_passive_raw[idx])
        
    w_active_arr = np.array(w_active_final)
    w_passive_arr = np.array(w_passive_final)
    
    if w_active_arr.sum() > 0: w_active_arr = w_active_arr / w_active_arr.sum()
    if w_passive_arr.sum() > 0: w_passive_arr = w_passive_arr / w_passive_arr.sum()
    
    returns = returns[final_assets]

    mean_daily = returns.mean().values
    cov_matrix = returns.cov().values + 1e-6 * np.eye(len(returns.columns))
    
    NUM_SIMS = 200
    DAYS = 252
    START_VAL = 100000
    
    sim_paths_active = np.zeros((DAYS, NUM_SIMS))
    sim_paths_passive = np.zeros((DAYS, NUM_SIMS))
    
    np.random.seed(42)
    
    for i in range(NUM_SIMS):
        try:
            daily_noise = np.random.multivariate_normal(mean_daily, cov_matrix, DAYS)
        except:
            daily_noise = np.random.normal(np.mean(mean_daily), np.mean(np.diag(cov_matrix)), (DAYS, len(mean_daily)))

        ret_active = np.dot(daily_noise, w_active_arr)
        sim_paths_active[:, i] = START_VAL * (1 + ret_active).cumprod()
        
        ret_passive = np.dot(daily_noise, w_passive_arr)
        sim_paths_passive[:, i] = START_VAL * (1 + ret_passive).cumprod()

    final_vals = sim_paths_active[-1, :]
    exp_ret = (final_vals.mean() - START_VAL) / START_VAL
    var_95 = START_VAL - np.percentile(final_vals, 5)
    
    final_vals_passive = sim_paths_passive[-1, :]
    exp_ret_passive = (final_vals_passive.mean() - START_VAL) / START_VAL
    
    avg_daily_ret = np.mean(sim_paths_active[1:] / sim_paths_active[:-1] - 1)
    std_daily_ret = np.std(sim_paths_active[1:] / sim_paths_active[:-1] - 1)
    sharpe = ((avg_daily_ret * 252) - 0.04) / (std_daily_ret * np.sqrt(252))
    
    avg_path = sim_paths_active.mean(axis=1)
    peak = np.maximum.accumulate(avg_path)
    drawdown = (avg_path - peak) / peak
    max_dd = drawdown.min()

    if is_dashboard:
        fig = go.Figure()
        for i in range(min(50, NUM_SIMS)):
            fig.add_trace(go.Scatter(
                y=sim_paths_active[:, i], mode='lines', 
                line=dict(color='rgba(0, 50, 150, 0.2)', width=1), showlegend=False))
            
        bench_path = sim_paths_passive.mean(axis=1)
        fig.add_trace(go.Scatter(y=bench_path, mode='lines', name='Strategic Benchmark', line=dict(color='#E74C3C', width=3)))
        fig.add_trace(go.Scatter(y=avg_path, mode='lines', name='ML-Enhanced Portfolio', line=dict(color='#0A2351', width=4), fill='tonexty', fillcolor='rgba(10, 35, 81, 0.1)'))
        
        fig.update_layout(
            title="Projected Performance (Active vs Passive)", xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)",
            margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        stats = {
            "Expected Return": exp_ret,
            "Benchmark Return": exp_ret_passive,
            "VaR 95%": var_95,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd
        }
        return fig, stats
    
    return None, None

if __name__ == "__main__":
    run_simulation(profile='Growth')