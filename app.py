import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh

# --- Streamlit UI ---
st.set_page_config(page_title="SPY Proxy MES Futures Trading Bot", layout="wide")
st.title("SPY Proxy MES Futures Trading Bot")
st.write("Live 1-min trading decisions based on VWAP + Supertrend + Truly Dynamic Bayesian Forecasting.")

# --- Parameters ---
symbol = st.text_input("Enter Symbol (default SPY):", value="SPY")
period = st.selectbox("Select period:", ["1d", "5d"], index=0)
refresh_rate = st.slider("Auto-refresh rate (seconds):", 30, 300, 60)
sensitivity = st.selectbox("Select Sensitivity:", ["Conservative", "Normal", "Aggressive"], index=1)

# --- Autorefresh every X seconds ---
st_autorefresh(interval=refresh_rate*1000, key="data_refresh")

# --- Functions ---
def calculate_bayesian_forecast(df, sensitivity_mode):
    atr_length = 10
    multiplier = 1.25
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(atr_length, min_periods=1).mean()

    hl2 = (df['High'] + df['Low']) / 2
    df['UpperBand'] = hl2 + (multiplier * df['ATR'])
    df['LowerBand'] = hl2 - (multiplier * df['ATR'])
    df['Supertrend'] = np.nan
    df['Direction'] = 0

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['UpperBand'].iloc[i-1]:
            df.at[df.index[i], 'Supertrend'] = df['LowerBand'].iloc[i]
            df.at[df.index[i], 'Direction'] = 1
        elif df['Close'].iloc[i] < df['LowerBand'].iloc[i-1]:
            df.at[df.index[i], 'Supertrend'] = df['UpperBand'].iloc[i]
            df.at[df.index[i], 'Direction'] = -1
        else:
            df.at[df.index[i], 'Supertrend'] = df['Supertrend'].iloc[i-1]
            df.at[df.index[i], 'Direction'] = df['Direction'].iloc[i-1]

    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    atr14 = df['TR'].rolling(14, min_periods=1).mean()
    atr_ma14 = atr14.rolling(14, min_periods=1).mean()
    vol_fact = atr14 / atr_ma14
    vol_fact.fillna(1, inplace=True)

    trend_strength = abs(df['Close'] - df['Supertrend']) / df['ATR']
    vwap_slope = df['VWAP'].diff(5)
    vwap_slope.fillna(0, inplace=True)

    combined_factor = (1/3) * (vol_fact.rank(pct=True) + trend_strength.rank(pct=True) + vwap_slope.abs().rank(pct=True))

    min_lb = 10
    max_lb = 60
    dyn_lb = (min_lb + (max_lb - min_lb) * (1 - combined_factor)).clip(lower=min_lb, upper=max_lb).astype(int)

    df['Mean'] = df['Close'].rolling(window=dyn_lb, min_periods=1).mean()
    df['Std'] = df['Close'].rolling(window=dyn_lb, min_periods=1).std()
    df['ZScore'] = (df['Close'] - df['Mean']) / df['Std']

    df['Prob_Up'] = norm.cdf(df['ZScore'])
    df['Prob_Down'] = 1 - df['Prob_Up']

    prior = 0.5
    den = (prior * df['Prob_Up']) + (prior * df['Prob_Down']) + 1e-6
    df['Posterior_Up'] = (prior * df['Prob_Up']) / den
    df['Posterior_Down'] = 1 - df['Posterior_Up']

    if sensitivity_mode == "Conservative":
        posterior_thresh = 0.95
    elif sensitivity_mode == "Aggressive":
        posterior_thresh = 0.85
    else:
        posterior_thresh = 0.9

    df['Buy_Signal'] = (df['Direction'] == -1) & (df['Posterior_Up'] > posterior_thresh)
    df['Sell_Signal'] = (df['Direction'] == 1) & (df['Posterior_Down'] > posterior_thresh)

    df['Background_Color'] = np.where(df['Buy_Signal'], 'green', np.where(df['Sell_Signal'], 'red', 'neutral'))

    return df

# --- Main Logic ---
placeholder = st.empty()

with placeholder.container():
    df = yf.download(tickers=symbol, interval="1m", period=period)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    df.columns = df.columns.str.replace(' ', '_').str.replace('__', '_')

    rename_map = {}
    for col in df.columns:
        if 'Open' in col: rename_map[col] = 'Open'
        if 'High' in col: rename_map[col] = 'High'
        if 'Low' in col: rename_map[col] = 'Low'
        if 'Close' in col and 'Adj' not in col: rename_map[col] = 'Close'
        if 'Volume' in col: rename_map[col] = 'Volume'
    df.rename(columns=rename_map, inplace=True)

    df = calculate_bayesian_forecast(df, sensitivity)

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    df['Signal'] = 0
    df.loc[(df['Buy_Signal']) & (df['Close'] > df['VWAP']) & (df['Close'] > df['EMA_20']), 'Signal'] = 1
    df.loc[(df['Sell_Signal']) & (df['Close'] < df['VWAP']) & (df['Close'] < df['EMA_20']), 'Signal'] = -1

    df = df.between_time('09:30', '16:00')

    df.dropna(inplace=True)

    features = ['Close', 'EMA_20', 'VWAP', 'ATR', 'ZScore']
    df['Future_Returns'] = df['Close'].shift(-5) - df['Close']
    df['Target'] = np.where(df['Future_Returns'] > 0, 1, 0)

    X = df[features]
    y = df['Target']

    model = RandomForestClassifier()
    model.fit(X[:-5], y[:-5])

    df['ML_Prediction'] = model.predict(X)

    df['Final_Signal'] = df.apply(lambda row: row['Signal'] if (row['Signal']==1 and row['ML_Prediction']==1) or (row['Signal']==-1 and row['ML_Prediction']==0) else 0, axis=1)

    starting_balance = 10000
    balance = starting_balance
    position = 0
    entry_price = 0
    profits = []
    trade_log = []

    for i in range(1, len(df)):
        if df['Final_Signal'].iloc[i] == 1 and position == 0:
            position = 1
            entry_price = df['Close'].iloc[i]
            trade_log.append((df.index[i], 'BUY', entry_price))
        elif df['Final_Signal'].iloc[i] == -1 and position == 0:
            position = -1
            entry_price = df['Close'].iloc[i]
            trade_log.append((df.index[i], 'SELL', entry_price))

        if position == 1 and (df['Close'].iloc[i] < df['VWAP'].iloc[i] or i == len(df)-1):
            pnl = df['Close'].iloc[i] - entry_price
            balance += pnl
            profits.append(pnl)
            position = 0

        if position == -1 and (df['Close'].iloc[i] > df['VWAP'].iloc[i] or i == len(df)-1):
            pnl = entry_price - df['Close'].iloc[i]
            balance += pnl
            profits.append(pnl)
            position = 0

    st.metric("Final Balance", f"${balance:.2f}")
    st.metric("Total Profit", f"${(balance - starting_balance):.2f}")
    st.metric("Number of Trades", len(profits))

    st.line_chart(df['Close'])
    st.line_chart(df['VWAP'])
    st.line_chart(df['EMA_20'])

    st.dataframe(df[['Close', 'EMA_20', 'VWAP', 'ATR', 'Supertrend', 'Posterior_Up', 'Posterior_Down', 'Signal', 'ML_Prediction', 'Final_Signal']].tail(50))

    trade_log_df = pd.DataFrame(trade_log, columns=['Time', 'Action', 'Price'])
    st.dataframe(trade_log_df)

    st.success("Trading Simulation Updated!")
