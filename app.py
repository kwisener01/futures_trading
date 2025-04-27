import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime, timedelta
import pytz
import requests

# --- Set page config ---
st.set_page_config(page_title="Futures Trading Bot", layout="wide")

# --- Helper Functions ---
def calculate_bayesian_forecast(df, sensitivity):
    atr_length = 10
    multiplier = 1.25

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(atr_length).mean()

    hl2 = (df['High'] + df['Low']) / 2
    df['UpperBand'] = hl2 + (multiplier * df['ATR'])
    df['LowerBand'] = hl2 - (multiplier * df['ATR'])

    df['Supertrend'] = 0.0
    df['Direction'] = 0

    for i in range(1, len(df)):
        curr_close = df['Close'].iloc[i]
        prev_supertrend = df['Supertrend'].iloc[i-1]

        if curr_close > df['UpperBand'].iloc[i]:
            df.at[df.index[i], 'Supertrend'] = df['LowerBand'].iloc[i]
            df.at[df.index[i], 'Direction'] = 1
        elif curr_close < df['LowerBand'].iloc[i]:
            df.at[df.index[i], 'Supertrend'] = df['UpperBand'].iloc[i]
            df.at[df.index[i], 'Direction'] = -1
        else:
            df.at[df.index[i], 'Supertrend'] = prev_supertrend
            df.at[df.index[i], 'Direction'] = df['Direction'].iloc[i-1]

    return df

# --- Sidebar ---
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Symbol (e.g., MES=F)", value="MES=F")
period = st.sidebar.selectbox("Period", options=['5d', '7d', '30d', '90d', '180d'], index=0)
sensitivity = st.sidebar.select_slider("Sensitivity", options=['aggressive', 'normal', 'conservative'], value='normal')
live_simulation = st.sidebar.checkbox("Live Simulation Mode", value=False)
use_alphavantage = st.sidebar.checkbox("Use AlphaVantage Feed", value=False)

# Use secret for API key
api_key = st.secrets.get("api_keys", {}).get("alphavantage", "")

# --- Main ---
st.title("🧠 Futures Trading Bot (Bayesian Forecast)")

if st.button("Start Trading Bot"):
    # Fetch Data
    if use_alphavantage and api_key:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        ts = data.get("Time Series (1min)", {})
        df = pd.DataFrame.from_dict(ts, orient='index')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
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

    # Apply Bayesian Forecast
    df = calculate_bayesian_forecast(df, sensitivity)

    # VWAP Calculation
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # EMA for Trend Filter
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Create ML Model
    df['Target'] = np.where(df['Close'].shift(-5) > df['Close'], 1, 0)
    X = df[['Close', 'EMA_20', 'Supertrend', 'VWAP', 'ATR']].fillna(0)
    y = df['Target']

    if len(X) > 5:
        model = RandomForestClassifier()
        model.fit(X[:-5], y[:-5])
        df['ML_Prediction'] = model.predict(X)
    else:
        df['ML_Prediction'] = 0

    # Trading Simulation
    balance = 10000
    open_trade = None
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        # Buy Signal
        if (row['ML_Prediction'] == 1) and (open_trade is None):
            atr = df['ATR'].iloc[i]
            tp_distance = atr * 1.5
            sl_distance = atr
            open_trade = {
                'Entry_Time': row.name,
                'Entry_Price': row['Close'],
                'Action': 'BUY',
                'TP_Price': row['Close'] + tp_distance,
                'SL_Price': row['Close'] - sl_distance
            }
            st.toast(f"📈 BUY Signal at {open_trade['Entry_Time']} {open_trade['Entry_Price']:.2f}")

        # Manage Open Trade
        if open_trade:
            if row['High'] >= open_trade['TP_Price']:
                pnl = open_trade['TP_Price'] - open_trade['Entry_Price']
                elapsed = row.name - open_trade['Entry_Time']
                balance += pnl
                trades.append({
                    'Entry_Time': open_trade['Entry_Time'],
                    'Action': open_trade['Action'],
                    'Entry_Price': open_trade['Entry_Price'],
                    'Exit_Time': row.name,
                    'Exit_Price': open_trade['TP_Price'],
                    'Elapsed': elapsed,
                    'PnL': pnl,
                    'Exit_Reason': 'TP'
                })
                st.toast(f"✅ TP Hit! Trade closed at {row.name} {open_trade['TP_Price']:.2f}")
                open_trade = None
            elif row['Low'] <= open_trade['SL_Price']:
                pnl = open_trade['SL_Price'] - open_trade['Entry_Price']
                elapsed = row.name - open_trade['Entry_Time']
                balance += pnl
                trades.append({
                    'Entry_Time': open_trade['Entry_Time'],
                    'Action': open_trade['Action'],
                    'Entry_Price': open_trade['Entry_Price'],
                    'Exit_Time': row.name,
                    'Exit_Price': open_trade['SL_Price'],
                    'Elapsed': elapsed,
                    'PnL': pnl,
                    'Exit_Reason': 'SL'
                })
                st.toast(f"🚨 SL Hit! Trade closed at {row.name} {open_trade['SL_Price']:.2f}")
                open_trade = None

    # Results
    st.subheader("Results")
    st.write(f"**Final Balance:** ${balance:.2f}")
    st.write(f"**Total Profit:** ${balance - 10000:.2f}")
    st.write(f"**Number of Trades:** {len(trades)}")

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['Elapsed'] = trades_df['Elapsed'].apply(lambda x: f"{x.components.minutes}m {x.components.seconds}s")
        st.dataframe(trades_df)

    # Plot
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df['Close'], label='Close Price')
    ax.plot(df['EMA_20'], label='EMA 20')
    ax.legend()
    ax.set_title(f"{symbol} Close Price with EMA20")
    st.pyplot(fig)

    if live_simulation:
        st.info("Live Simulation Mode Active - Trades will be printed here live when detected.")
