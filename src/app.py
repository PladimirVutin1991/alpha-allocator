import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
from portfolio_simulation import run_simulation, ALLOCATIONS 

TICKER_TO_NAME = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'ORCL': 'Oracle', 'ADBE': 'Adobe',
    'GOOGL': 'Alphabet', 'META': 'Meta', 'JPM': 'JPMorgan', 'BAC': 'Bank of America',
    'LLY': 'Eli Lilly', 'UNH': 'UnitedHealth', 'AMZN': 'Amazon', 'TSLA': 'Tesla',
    'XOM': 'Exxon Mobil', 'CVX': 'Chevron', 'BTC-USD': 'Bitcoin', 'AGG': 'Bonds', 'SHV': 'Cash',
    'SPY': 'S&P 500 ETF', 'GLD': 'Gold', 'VNQ': 'Real Estate ETF', 'DBC': 'Commodities'
}

st.set_page_config(page_title="Alpha Allocator (ISL)", layout="wide")
st.title("🏛️ Alpha Allocator (ISL)")

st.sidebar.header("Client Mandate")
risk_label = st.sidebar.selectbox("Risk Profile", ["Low Risk", "Balanced Risk", "High Risk"])
RISK_MAPPING = {"High Risk": "Growth", "Balanced Risk": "Balanced", "Low Risk": "Income"}
profile = RISK_MAPPING[risk_label]

threshold = st.sidebar.slider(
    "Conviction Threshold", 0.50, 0.80, 0.53, 0.01,
    help="Minimum AI confidence score required to execute a trade."
)
st.sidebar.markdown("---")

st.sidebar.subheader("Base Allocation")
weights = ALLOCATIONS[profile]
labels = ["Equities", "Fixed Income", "Liquidity", "Alternatives"]
values = [weights['Equity'], weights['Fixed_Income'], weights['Cash'], weights['Alts']]
colors = ['#0A2351', '#C5A065', '#E5E7E9', '#4B0082']

fig_pie = go.Figure(data=[go.Pie(
    labels=labels, values=values, hole=.4, 
    marker=dict(colors=colors), textinfo='percent', hoverinfo='label+percent',      
    sort=False, direction='clockwise', rotation=90            
)])
fig_pie.update_layout(
    showlegend=True, legend=dict(orientation="h", y=-0.3), 
    margin=dict(t=0, b=0, l=0, r=0), height=250, 
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
)
st.sidebar.plotly_chart(fig_pie, use_container_width=True)

with st.spinner(f"Calibrating {risk_label} Portfolio..."):
    try:
        fig_sim, sim_stats = run_simulation(profile, threshold, is_dashboard=True)
    except Exception as e:
        st.error(f"Simulation Error: {e}")
        fig_sim, sim_stats = None, None

tab1, tab2, tab3 = st.tabs(["📋 Portfolio & Stock Picks", "⚖️ Simulation Comparison", "🔍 Model Validation"])

if os.path.exists('data/model_predictions.csv'):
    df = pd.read_csv('data/model_predictions.csv')
    buys = df[(df['Predicted_Signal'] == 1) & (df['Confidence_Score'] > threshold)]
    
    if not buys.empty:
        unique_picks = buys.sort_values('Date').groupby('Ticker').tail(1)
        unique_picks = unique_picks.sort_values(by='Confidence_Score', ascending=False)
        unique_picks['Asset Name'] = unique_picks['Ticker'].apply(lambda x: TICKER_TO_NAME.get(x, x))
    else:
        unique_picks = pd.DataFrame(columns=['Ticker', 'Asset Name', 'Confidence_Score'])

    with tab1:
        st.subheader(f"Proposed {risk_label} Portfolio")
        col_metrics, col_picks, col_sector = st.columns([1, 1.5, 1.5])
        
        with col_metrics:
            st.markdown("#### Key Metrics")
            if sim_stats:
                st.metric("Proj. Return", f"{sim_stats['Expected Return']:.1%}", delta="vs Benchmark")
                st.metric("Potential Loss (VaR)", f"${abs(sim_stats['VaR 95%']):,.0f}", delta_color="inverse")
                st.markdown("---")
                st.metric("Sharpe Ratio", f"{sim_stats['Sharpe Ratio']:.2f}", help="Risk-Adjusted Return (>1 is good)")
                st.metric("Max Drawdown", f"{sim_stats['Max Drawdown']:.1%}", help="Worst projected drop from peak", delta_color="inverse")
            else:
                st.warning("Metrics unavailable.")

        with col_picks:
            st.markdown("#### Tactical Picks (Alpha)")
            if not unique_picks.empty:
                display_df = unique_picks[['Ticker', 'Asset Name', 'Confidence_Score']].copy()
                display_df['Confidence_Score'] = display_df['Confidence_Score'] * 100 
                styled_df = display_df.style.background_gradient(subset=['Confidence_Score'], cmap='Greens', vmin=50, vmax=70).format({'Confidence_Score': '{:.2f}%'})
                st.dataframe(styled_df, column_config={"Ticker": st.column_config.TextColumn("Ticker", width="small"), "Asset Name": st.column_config.TextColumn("Asset Name", width="medium"), "Confidence_Score": st.column_config.NumberColumn("Conviction")}, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No Alpha Picks found.")

        with col_sector:
            st.markdown("#### Active Sector Exposure")
            if not unique_picks.empty:
                if 'SEC_' in str(df.columns): 
                    sector_cols = [c for c in df.columns if 'SEC_' in c]
                    pick_sectors = df[df['Ticker'].isin(unique_picks['Ticker'])].groupby('Ticker').tail(1)
                    pick_sectors['Sector'] = pick_sectors[sector_cols].idxmax(axis=1).str.replace('SEC_', '').str.replace('_', ' ')
                    
                    sec_counts = pick_sectors['Sector'].value_counts().reset_index()
                    sec_counts.columns = ['Sector', 'Count']
                    fig_sec_donut = px.pie(sec_counts, values='Count', names='Sector', hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig_sec_donut.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, showlegend=True)
                    fig_sec_donut.update_traces(hovertemplate='%{label}: %{value} stocks<extra></extra>')
                    st.plotly_chart(fig_sec_donut, use_container_width=True)
            else:
                st.info("No active exposure.")

    with tab2:
        st.subheader("Performance Projection: ML-Enhanced vs. Benchmark")
        if fig_sim and sim_stats:
            final_val = 100000 * (1 + sim_stats['Expected Return'])
            pct_return = sim_stats['Expected Return']
            final_val_bench = 100000 * (1 + sim_stats['Benchmark Return'])
            pct_return_bench = sim_stats['Benchmark Return']
            
            # --- BOXES ---
            fig_sim.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,1)', 
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'), yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                font=dict(color="black"), dragmode='pan', height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                annotations=[
                    # Blue Box (ML)
                    dict(x=1, y=0.9, xref="paper", yref="paper", 
                         text=f"<b>Expected Value (1Y):</b><br>${final_val:,.0f}<br>(+{pct_return:.1%})", 
                         showarrow=False, font=dict(size=12, color="#0A2351"), 
                         bgcolor="rgba(255,255,255,0.9)", bordercolor="#0A2351", borderwidth=1),
                    # Red Box (Benchmark) 
                    dict(x=1, y=0.75, xref="paper", yref="paper", 
                         text=f"<b>Benchmark (1Y):</b><br>${final_val_bench:,.0f}<br>(+{pct_return_bench:.1%})", 
                         showarrow=False, font=dict(size=12, color="#E74C3C"), 
                         bgcolor="rgba(255,255,255,0.9)", bordercolor="#E74C3C", borderwidth=1)
                ]
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.error("Simulation failed.")

    with tab3:
        st.subheader("Model Accuracy Audit")
        conditions = [
            (df['Predicted_Signal'] == 1) & (df['Target'] == 1), 
            (df['Predicted_Signal'] == 1) & (df['Target'] == 0),
            (df['Predicted_Signal'] == 0) & (df['Target'] == 1), 
            (df['Predicted_Signal'] == 0) & (df['Target'] == 0)
        ]
        df['Outcome'] = np.select(conditions, ['Win', 'Loss', 'Missed', 'Correct Avoid'], default='Error')
        
        col_acc1, col_acc2 = st.columns(2)
        with col_acc1:
            st.markdown("##### Sector Accuracy")
            sector_cols = [c for c in df.columns if 'SEC_' in c]
            if sector_cols:
                df['Sector'] = df[sector_cols].idxmax(axis=1).str.replace('SEC_', '').str.replace('_', ' ')
                buys_only = df[df['Predicted_Signal'] == 1]
                if not buys_only.empty:
                    sec_stats = buys_only.groupby('Sector').apply(lambda x: (x['Target'] == 1).mean()).reset_index(name='Accuracy')
                    fig_bar = px.bar(sec_stats, x='Sector', y='Accuracy', color='Accuracy', color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[0.40, 0.60], text_auto='.1%')
                    fig_bar.update_layout(xaxis_title=None, yaxis_title="Win Rate")
                    fig_bar.add_hline(y=0.5, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.info("Precision by Sector: Represents the percentage of correctly identified outperforming stocks (Win Rate) within each sector subset.")
                else:
                    st.info("No trades to analyze.")

        with col_acc2:
            st.markdown("##### Outcome Distribution")
            outcome_counts = df['Outcome'].value_counts().reset_index()
            outcome_counts.columns = ['Outcome', 'Count']
            active_counts = outcome_counts[outcome_counts['Outcome'].isin(['Win', 'Loss'])]
            if not active_counts.empty:
                fig_donut = px.pie(active_counts, values='Count', names='Outcome', color='Outcome', color_discrete_map={'Win': '#2ECC71', 'Loss': '#E74C3C'}, hole=0.6)
                fig_donut.update_traces(hovertemplate='%{label}: %{value} trades<extra></extra>')
                st.plotly_chart(fig_donut, use_container_width=True)
                st.info("Active Trade Outcomes: 'Win' indicates a predicted buy that outperformed the benchmark; 'Loss' indicates a predicted buy that underperformed.")
            else:
                st.info("No active trades.")

else:
    st.error("⚠️ Predictions missing. Please run 'python src/optimize_model.py' first.")