import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="BTC 4H Predictor", layout="wide", page_icon="🚀")
st.title("🚀 Bitcoin 4-Hour Alpha Predictor")

@st.cache_resource
def load_assets():
    with open('btc_production_model.pkl', 'rb') as f:
        return pickle.load(f)

# Load variables into memory
prod = load_assets()
boosters = prod['boosters']
feature_names = prod['feature_names']

# --- 2. DATA FETCHING ---
def get_live_data():
    # Attempt Binance, fallback to OKX
    try:
        ex = ccxt.binance({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv('BTC/USDT', '15m', limit=100)
    except:
        ex = ccxt.okx({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv('BTC/USDT', '15m', limit=100)
    
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df['instrument'] = 'BTC'
    return df.set_index(['datetime', 'instrument'])

# --- 3. FACTOR CALCULATION ---
def calculate_factors(df):
    d = df.copy()
    feat = pd.DataFrame(index=d.index)
    # Manual Technical Factors
    feat['KMID'] = (d['close'] - d['open']) / d['open']
    feat['KLEN'] = (d['high'] - d['low']) / d['open']
    feat['KMID2'] = (d['close'] - d['open']) / (d['high'] - d['low'] + 1e-12)
    for w in [5, 10, 20, 30, 60]:
        feat[f'ROC{w}'] = d['close'].shift(w) / d['close'] - 1
        feat[f'MA{w}'] = d['close'].rolling(w).mean() / d['close']
        feat[f'STD{w}'] = d['close'].rolling(w).std() / d['close']
        feat[f'VMA{w}'] = d['volume'].rolling(w).mean() / (d['volume'] + 1e-12)
        feat[f'CORR{w}'] = d['close'].rolling(w).corr(np.log1p(d['volume']))
    
    # Proactive Filter: Only use features present in the trained model
    # (Excludes 'label' if it was saved in the names list)
    valid_cols = [c for c in feature_names if c in feat.columns]
    return feat[valid_cols].ffill().bfill()

# --- 4. INFERENCE ENGINE ---
def run_prediction():
    # Get raw data
    data_raw = get_live_data()
    # Calculate factors
    feats = calculate_factors(data_raw)
    
    # Scale Data (X - Mean) / Std
    # We use .values to ensure alignment with the numpy boosters
    # We ensure Mean/Std match the exact columns being used
    m = pd.Series(prod['mean'], index=feature_names)[feats.columns].values
    s = pd.Series(prod['std'], index=feature_names)[feats.columns].values
    
    x_scaled = (feats.values - m) / (s + 1e-12)
    x_scaled = np.clip(x_scaled, -3, 3)
    
    # Ensemble Prediction
    preds = []
    for booster in boosters:
        preds.append(booster.predict(x_scaled))
    
    final_signal = np.mean(preds, axis=0)
    
    return data_raw, final_signal[-1]

# --- 5. USER INTERFACE ---
try:
    # Run the full pipeline
    data, latest_sig = run_prediction()
    
    current_price = data['close'].iloc[-1]
    target_price = current_price * (1 + latest_sig)
    last_ts = data.index.get_level_values(0)[-1]
    target_time = (last_ts + timedelta(hours=4)).strftime('%H:%M')

    # Top Metric Bar
    col1, col2, col3 = st.columns(3)
    col1.metric("Live BTC Price", f"${current_price:,.2f}")
    col2.metric("4H Prediction", f"{latest_sig*100:.3f}%", delta=f"{latest_sig*100:.4f}%")
    col3.metric(f"Target (at {target_time} UTC)", f"${target_price:,.2f}")

    # Candlestick Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index.get_level_values(0),
        open=data['open'], high=data['high'],
        low=data['low'], close=data['close'], name='BTC'
    ))

    # Projection Line
    fig.add_trace(go.Scatter(
        x=[last_ts, last_ts + timedelta(hours=4)],
        y=[current_price, target_price],
        line=dict(color='orange', width=4, dash='dot'),
        name="4H Forecast"
    ))

    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Last data point: {last_ts} UTC. Prediction logic: DoubleEnsemble (6 models).")

except Exception as e:
    st.error(f"Inference Engine Error: {e}")

if st.button('🔄 Refresh Real-Time Data'):
    st.rerun()
