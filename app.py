import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. SETUP & LOAD ---
st.set_page_config(page_title="BTC 4H Predictor", layout="wide", page_icon="🚀")
st.title("🚀 Bitcoin 4-Hour Alpha Predictor")

@st.cache_resource
def load_production_model():
    with open('btc_production_model.pkl', 'rb') as f:
        return pickle.load(f)

prod = load_production_model()

# --- 2. DATA & MATH ---
def calculate_factors(df):
    d = df.copy()
    feat = pd.DataFrame(index=d.index)
    feat['KMID'] = (d['close'] - d['open']) / d['open']
    feat['KLEN'] = (d['high'] - d['low']) / d['open']
    feat['KMID2'] = (d['close'] - d['open']) / (d['high'] - d['low'] + 1e-12)
    for w in [5, 10, 20, 30, 60]:
        feat[f'ROC{w}'] = d['close'].shift(w) / d['close'] - 1
        feat[f'MA{w}'] = d['close'].rolling(w).mean() / d['close']
        feat[f'STD{w}'] = d['close'].rolling(w).std() / d['close']
        feat[f'VMA{w}'] = d['volume'].rolling(w).mean() / (d['volume'] + 1e-12)
        feat[f'CORR{w}'] = d['close'].rolling(w).corr(np.log1p(d['volume']))
    
    # PROACTIVE FIX: Only use columns that exist in BOTH our math and the model's memory
    # This filters out 'label' or '4h_return' from the required list
    valid_features = [f for f in prod['feature_names'] if f in feat.columns]
    return feat[valid_features].ffill().bfill()

# --- 3. INFERENCE ---
def run_prediction():
    data_df = get_live_data()
    feats = calculate_factors(data_df)
    
    # PROACTIVE FIX: Align the Mean and Std stats to match our actual features
    # This prevents 'Label' from poisoning the subtraction math
    model_mean = prod['mean'][feats.columns]
    model_std = prod['std'][feats.columns]
    
    # Scaling
    x_scaled = (feats - model_mean) / (model_std + 1e-12)
    x_scaled = x_scaled.clip(-3, 3)
    
    all_preds = []
    for booster in prod['boosters']:
        all_preds.append(booster.predict(x_scaled.values))
    
    final_score = np.mean(all_preds, axis=0)
    return data_df, final_score[-1]

# --- 4. UI ---
try:
    data, latest_sig = run_prediction()
    current_price = data['close'].iloc[-1]
    target_price = current_price * (1 + latest_sig)
    last_ts = data.index[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Live Price", f"${current_price:,.2f}")
    col2.metric("4H Predicted Return", f"{latest_sig*100:.3f}%")
    col3.metric("Target Price (UTC {0})".format((last_ts + timedelta(hours=4)).strftime('%H:%M')), f"${target_price:,.2f}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], 
                                 low=data['low'], close=data['close'], name="BTC"))
    fig.add_trace(go.Scatter(x=[last_ts, last_ts + timedelta(hours=4)], y=[current_price, target_price],
                             line=dict(color='cyan', width=4, dash='dot'), name="4H Projection"))

    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Last updated: {last_ts} UTC. Prediction covers the following 16 candles (4 hours).")

except Exception as e:
    st.warning(f"Connecting to exchange... ({e})")

if st.button('🔄 Refresh Prediction'):
    st.rerun()
