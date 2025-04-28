import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import pytz
import requests
from streamlit_autorefresh import st_autorefresh
import os

# --- Set page config ---
st.set_page_config(page_title="Futures Trading Bot", layout="wide")

# --- Helper Functions ---
def fetch_alpaca_data(symbol, start, end, timeframe):
    base_url = "https://data.alpaca.markets/v2/stocks"
    headers = {
        "APCA-API-KEY-ID": st.secrets["ALPACA"]["API_KEY"],
        "APCA-API-SECRET-KEY": st.secrets["ALPACA"]["SECRET_KEY"],
    }
    params = {"start": start, "end": end, "timeframe": timeframe, "limit": 10000}
    url = f"{base_url}/{symbol}/bars"
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()['bars']
    df = pd.DataFrame(data)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
    return df

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
use_alpaca = st.sidebar.checkbox("Use Alpaca Live Feed", value=False)
refresh_minutes = st.sidebar.number_input("Refresh Interval (minutes)", min_value=1, max_value=30, value=5)

# --- Main ---
st.title("\ud83e\uddd1\u200d\ud83d\udcbb Futures Trading Bot (Bayesian Forecast)")

# --- Bot Execution ---
def run_bot():
    global symbol, use_alpaca

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{symbol.replace('=F','')}_live_1m_data.csv")

    try:
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
    except Exception as e:
        st.error(f"Data fetch failed. Switching to SPY on Yahoo Finance. ({str(e)})")
        symbol = "SPY"
        use_alpaca = False
        df = yf.download(tickers=symbol, interval="1m", period=period)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df.tz_convert('US/Eastern')

    if os.path.exists(csv_path) and live_simulation:
        old_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = pd.concat([old_df, df]).drop_duplicates().sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.columns = df.columns.str.replace(' ', '_').str.replace('__', '_')

    rename_map = {col: col.split('_')[0] for col in df.columns}
    df.rename(columns=rename_map, inplace=True)

    df = calculate_bayesian_forecast(df, sensitivity)

    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    df['Target'] = np.where(df['Close'].shift(-5) > df['Close'], 1, 0)
    X = df[['Close', 'EMA_20', 'Supertrend', 'VWAP', 'ATR']].fillna(0)
    y = df['Target']

    if len(X) > 5:
        model = RandomForestClassifier()
        model.fit(X[:-5], y[:-5])
        df['ML_Prediction'] = model.predict(X)
    else:
        df['ML_Prediction'] = 0

    balance = 10000
    open_trade = None
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]

        if (row['ML_Prediction'] == 1) and (open_trade is None):
            open_trade = {
                'Entry_Time': row.name,
                'Entry_Price': row['Close'],
                'Action': 'BUY',
                'TP_Price': row['Close'] * 1.002,
                'SL_Price': row['Close'] * 0.998
            }
            st.toast(f"\ud83d\udcc8 BUY Signal at {open_trade['Entry_Time']} {open_trade['Entry_Price']:.2f}")

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
                st.toast(f"\u2705 TP Hit at {row.name} {open_trade['TP_Price']:.2f}")
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
                st.toast(f"\ud83d\udea8 SL Hit at {row.name} {open_trade['SL_Price']:.2f}")
                open_trade = None

    st.subheader("Results")
    st.write(f"**Final Balance:** ${balance:.2f}")
    st.write(f"**Total Profit:** ${balance - 10000:.2f}")
    st.write(f"**Number of Trades:** {len(trades)}")

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['Elapsed'] = trades_df['Elapsed'].apply(lambda x: f"{x.components.minutes}m {x.components.seconds}s")
        trades_df = trades_df.style.applymap(lambda v: 'color: green;' if isinstance(v, (int, float)) and v > 0 else 'color: red;', subset=['PnL'])
        st.dataframe(trades_df)
    else:
        st.info("Waiting for new trades...")

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df['Close'], label='Close Price')
    ax.plot(df['EMA_20'], label='EMA 20')
    ax.legend()
    ax.set_title(f"{symbol} Close Price with EMA20")
    st.pyplot(fig)

    # Save updated data
    df.to_csv(csv_path)
    st.caption(f"Live 1-minute data saved: {csv_path}")

    # Download CSV Button
    st.download_button(
        label="Download 1-min Live Data",
        data=df.to_csv().encode('utf-8'),
        file_name=f"{symbol.replace('=F','')}_live_data.csv",
        mime='text/csv'
    )

if st.button("Start Trading Bot") or live_simulation:
    run_bot()
    if live_simulation:
        st_autorefresh(interval=refresh_minutes * 60 * 1000, key="live_refresh")
