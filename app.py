import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pytz
import requests
import time

# --- Set page config ---
st.set_page_config(page_title="Futures Trading Bot", layout="wide")

# --- Helper Functions ---
def fetch_alpaca_data(symbol, start, end, timeframe):
    base_url = "https://data.alpaca.markets/v2/stocks"
    headers = {
        "APCA-API-KEY-ID": st.secrets["ALPACA"]["API_KEY"],
        "APCA-API-SECRET-KEY": st.secrets["ALPACA"]["SECRET_KEY"],
    }
    params = {
        "start": start,
        "end": end,
        "timeframe": timeframe,
        "limit": 10000,
    }
    url = f"{base_url}/{symbol}/bars"
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()['bars']
    df = pd.DataFrame(data)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
    return df

# --- Sidebar ---
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol (e.g., MES=F)", value="MES=F")
period = st.sidebar.selectbox("Period", options=['5d', '7d', '30d', '90d', '180d'], index=0)
sensitivity = st.sidebar.select_slider("Sensitivity", options=['aggressive', 'normal', 'conservative'], value='normal')
live_simulation = st.sidebar.checkbox("Live Simulation Mode", value=False)
use_alpaca = st.sidebar.checkbox("Use Alpaca Live Feed", value=False)
refresh_rate = st.sidebar.number_input("Refresh Rate (minutes)", min_value=1, max_value=60, value=5)

# --- Main ---
st.title("🧠 Futures Trading Bot (Bayesian Forecast)")

def trading_bot():
    if use_alpaca:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=int(period.replace('d', '')))
        df = fetch_alpaca_data(symbol, start_date.isoformat() + 'Z', end_date.isoformat() + 'Z', "1Min")
        df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    else:
        df = yf.download(tickers=symbol, interval="1m", period=period)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df.tz_convert('US/Eastern')

    # Clean Columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.columns = df.columns.str.replace(' ', '_').str.replace('__', '_')

    rename_map = {}
    for col in df.columns:
        if 'Open' in col: rename_map[col] = 'Open'
        if 'High' in col: rename_map[col] = 'High'
        if 'Low' in col: rename_map[col] = 'Low'
        if 'Close' in col: rename_map[col] = 'Close'
        if 'Volume' in col: rename_map[col] = 'Volume'
    df.rename(columns=rename_map, inplace=True)

    # --- Example trading logic ---
    st.toast("✅ Data Loaded Successfully!", icon="📈")

if st.button("Start Trading Bot") or live_simulation:
    trading_bot()

    if live_simulation:
        while True:
            time.sleep(refresh_rate * 60)
            st.experimental_rerun()
