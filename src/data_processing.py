import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. CONFIGURATION
# ==========================================

START_DATE = '2015-01-01' 
END_DATE = '2025-12-30'
PREDICTION_HORIZON = 63 

# --- CLEANED TICKERS ---
SECTOR_LISTS = {
    'Tech': [
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'INTC', 'TXN', 'CSCO', 'ACN', 'IBM', 'NOW', 
        'UBER', 'PANW', 'PLTR', 'SNOW', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'ROP', 'NXPI', 'FTNT', 'APH', 'ADI',
        'TEL', 'HPQ', 'GLW', 'KEYS', 'IT', 'CDW', 'BR', 'TRMB', 'TYL', 'PTC', 'AKAM', 'GEN', 'STX', 'NTAP'
    ],
    'Financials': [
        'JPM', 'BAC', 'WFC', 'C', 'MS', 'GS', 'BLK', 'SCHW', 'AXP', 'SPGI', 'PGR', 'CB', 'MMC', 'V', 'MA', 'PYPL', 
        'COF', 'USB', 'PNC', 'TFC', 'BK', 'ICE', 'CME', 'MCO', 'AON', 'AJG', 'TRV', 'AFL', 'ALL', 'HIG', 'MET', 'PRU',
        'AMP', 'STT', 'NTRS', 'FITB', 'RF', 'HBAN', 'KEY', 'CFG', 'SYF', 'JKHY', 'FISV', 'GPN'
    ],
    'Healthcare': [
        'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'AMGN', 'PFE', 'ISRG', 'DHR', 'SYK', 'ELV', 'GILD', 'BMY', 'CVS', 
        'CI', 'REGN', 'VRTX', 'ZTS', 'BDX', 'BSX', 'HUM', 'MCK', 'COR', 'HCA', 'EW', 'CNC', 'IQV', 'A', 'RMD', 'MTD',
        'RVTY', 'TFX', 'COO', 'WAT', 'HOLX', 'BAX', 'STE', 'DGX', 'LH', 'ALGN', 'XRAY', 'BIO', 'TECH', 'WST'
    ],
    'Consumer': [
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'BKNG', 'TJX', 'TGT', 'WMT', 'PG', 'COST', 'KO', 'PEP', 
        'PM', 'MO', 'EL', 'CL', 'DG', 'DLTR', 'MNST', 'KDP', 'K', 'GIS', 'HSY', 'KMB', 'SYY', 'ADM', 'STZ', 'TSCO',
        'ROST', 'ORLY', 'AZO', 'ULTA', 'BBY', 'GPC', 'LKQ', 'DPZ', 'YUM', 'CMG', 'DRI', 'DHI', 'LEN', 'PHM', 'NVR'
    ],
    'Energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI', 'WMB', 'OKE', 'TRGP', 'HAL', 
        'BKR', 'DVN', 'CTRA', 'APA', 'EQT' 
    ],
    'Industrials': [
        'GE', 'CAT', 'UNP', 'HON', 'UPS', 'BA', 'LMT', 'RTX', 'DE', 'MMM', 'ETN', 'ITW', 'WM', 'CSX', 'NSC', 'FDX',
        'GD', 'NOC', 'LHX', 'TDG', 'TXT', 'HII', 'EMR', 'PH', 'TT', 'CARR', 'OTIS', 'ROK', 'AME', 'FAST', 'PCAR'
    ],
    'Utilities': [
        'NEE', 'DUK', 'SO', 'AEP', 'ED', 'D', 'SRE', 'PEG', 'WEC', 'XEL', 'ES', 'DTE', 'FE', 'PPL', 'EIX', 'AEE', 
        'CMS', 'CNP', 'ATO', 'EVRG', 'LNT', 'NI', 'PNW'
    ],
    'Real Estate': [
        'PLD', 'AMT', 'CCI', 'PSA', 'O', 'DLR', 'SPG', 'WELL', 'EQIX', 'SBAC', 'VICI', 'AVB', 'EQR', 'EXR', 'MAA',
        'ESS', 'UDR', 'IRM', 'KIM', 'REG', 'FRT', 'HST'
    ],
    'Materials': [
        'LIN', 'SHW', 'FCX', 'NEM', 'APD', 'ECL', 'CTVA', 'DOW', 'DD', 'PPG', 'ALB', 'FMC', 'MOS', 'CF', 'NUE',
        'STLD', 'VMC', 'MLM', 'BALL', 'AMCR', 'PKG'
    ],
    'Comm Services': [
        'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'WBD', 'OMC', 'IPG', 'LYV',
        'EA', 'TTWO', 'MTCH'
    ]
}

STOCK_UNIVERSE = [ticker for sector in SECTOR_LISTS.values() for ticker in sector]

MACRO_ASSETS = {
    'SPY': 'Equity_Benchmark',
    'AGG': 'Bonds_Core',
    'LQD': 'Bonds_Corp',
    'HYG': 'Bonds_HighYield',
    'SHV': 'Cash',
    'GLD': 'Alts_Gold',
    'DBC': 'Alts_Commodities',
    'VNQ': 'Alts_RealEstate'
}

def get_sector(ticker):
    for sector, tickers in SECTOR_LISTS.items():
        if ticker in tickers: return sector
    return 'Other'

def safe_download(tickers, start, end):
    print(f"Downloading {len(tickers)} assets...")
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty: return pd.DataFrame()
        if 'Adj Close' in data.columns: return data['Adj Close']
        if 'Close' in data.columns: return data['Close']
        if isinstance(data.columns, pd.MultiIndex):
            levels = data.columns.get_level_values(0)
            if 'Adj Close' in levels: return data['Adj Close']
            if 'Close' in levels: return data['Close']
        return pd.DataFrame()
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def get_market_caps(tickers):
    print("--- Fetching Market Caps ---")
    mcaps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mcaps[t] = info.get('marketCap', 10e9)
        except: mcaps[t] = 10e9
    return mcaps

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_technical_features(price_series, benchmark_series):
    df = pd.DataFrame(index=price_series.index)
    price_series = price_series.ffill()
    benchmark_series = benchmark_series.ffill()

    df['Ret_1M'] = price_series.pct_change(21)
    df['Ret_3M'] = price_series.pct_change(63)
    df['RSI'] = calculate_rsi(price_series)
    sma_50 = price_series.rolling(50).mean()
    df['Trend_SMA'] = (price_series / sma_50) - 1
    
    return df

def prepare_data():
    if not os.path.exists('data'): os.makedirs('data')

    all_tickers = STOCK_UNIVERSE + list(MACRO_ASSETS.keys())
    prices = safe_download(all_tickers, START_DATE, END_DATE)
    prices = prices.ffill()
    
    stock_prices = prices[[t for t in STOCK_UNIVERSE if t in prices.columns]]
    macro_prices = prices[[t for t in MACRO_ASSETS.keys() if t in prices.columns]]
    
    stock_prices.to_csv('data/stock_prices.csv')
    macro_prices.to_csv('data/macro_prices.csv')

    if not os.path.exists('data/market_caps.csv'):
        mcaps = get_market_caps(stock_prices.columns.tolist())
        pd.Series(mcaps).to_csv('data/market_caps.csv')

    print("--- Building Features ---")
    ml_data_list = []
    
    if 'SPY' not in macro_prices.columns:
        print("❌ Error: SPY missing.")
        return

    spy = macro_prices['SPY']

    for ticker in stock_prices.columns:
        try:
            series = stock_prices[ticker]
            feats = calculate_technical_features(series, spy)
            feats['Sector'] = get_sector(ticker)
            feats['Ticker'] = ticker
            feats['Future_Ret'] = series.shift(-PREDICTION_HORIZON) / series - 1
            ml_data_list.append(feats.dropna())
        except: continue

    if ml_data_list:
        full_df = pd.concat(ml_data_list)
        print("Calculating Sector-Relative Targets...")
        sector_medians = full_df.groupby(['Date', 'Sector'])['Future_Ret'].transform('median')
        full_df['Target_Return_Beat'] = (full_df['Future_Ret'] > sector_medians).astype(int)

        full_df = pd.get_dummies(full_df, columns=['Sector'], prefix='SEC')
        full_df = full_df[full_df.index >= START_DATE] 
        
        full_df.to_csv('data/ml_dataset.csv')
        print(f"✅ ML Dataset saved. Shape: {full_df.shape}")
        print(f"Class Balance: {full_df['Target_Return_Beat'].mean():.2%}")
    else:
        print("❌ Failed.")

if __name__ == "__main__":
    prepare_data()