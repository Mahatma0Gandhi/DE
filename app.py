import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# --- 1. SETUP ---
st.set_page_config(page_title="BTC Alpha Live", layout="wide")

@st.cache_resource
def load_assets():
    with open('btc_production_model.pkl', 'rb') as f:
        return pickle.load(f)

prod = load_assets()

# --- 2. THE HARDENED ENGINE ---
def get_live_data():
    # Force fresh data by using a timestamp parameter
    ex = ccxt.okx()
    ohlcv = ex.fetch_ohlcv('BTC/USDT', '15m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    # Use only datetime as index for simplicity and sort it!
    df = df.set_index('datetime').sort_index()
    return df

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
    
    # Ensure column alignment with production model
    valid_cols = [c for c in prod['feature_names'] if c in feat.columns]
    return feat[valid_cols].ffill().bfill()

def run_prediction():
    data_df = get_live_data()
    feats = calculate_factors(data_df)
    
    # Scale Data
    m = pd.Series(prod['mean'], index=prod['feature_names'])[feats.columns].values
    s = pd.Series(prod['std'], index=prod['feature_names'])[feats.columns].values
    x_scaled = np.clip((feats.values - m) / (s + 1e-12), -3, 3)
    
    # Ensemble Prediction
    all_preds = []
    for booster in prod['boosters']:
        all_preds.append(booster.predict(x_scaled))
    
    # This results in a 1D array of signals for the last 200 candles
    final_signals = np.mean(all_preds, axis=0)
    
    # Create a clean signal series
    sig_series = pd.Series(final_signals, index=data_df.index)
    return data_df, sig_series

# --- 3. UI ---
st.title("🚀 Bitcoin 4-Hour Alpha Predictor")

try:
    data, sigs = run_prediction()
    
    # GRAB LATEST
    curr_p = data['close'].iloc[-1]
    latest_sig = sigs.iloc[-1]
    target_p = curr_p * (1 + latest_sig)
    last_ts = data.index[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Live Price", f"${curr_p:,.2f}")
    # Delta shows change from previous 15m candle
    col2.metric("4H Prediction", f"{latest_sig*100:.4f}%", delta=f"{(latest_sig - sigs.iloc[-2])*100:.4f}%")
    col3.metric("Target Price (4H)", f"${target_p:,.2f}")

    # THE "SMOKING GUN" TABLE
    st.subheader("📊 Live Signal Momentum (Compare to Colab)")
    # Show the last 5 signals like your Colab log
    history_df = pd.DataFrame({
        'Price': data['close'],
        'Signal %': sigs * 100
    }).tail(10)
    st.table(history_df.style.format({'Price': '{:,.2f}', 'Signal %': '{:,.4f}%'}))

    # FOOTER
    st.caption(f"Last sync: {last_ts} UTC | Dashboard Time: {datetime.now().strftime('%H:%M:%S')} IST")

except Exception as e:
    st.error(f"Engine Error: {e}")

if st.button('🔄 Sync Data Now'):
    st.rerun()
