import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

def compute_moving_averages(df, window=50):
    df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
    return df

def compute_volatility(df, window=50):
    df[f"Volatility_{window}"] = df["Close"].pct_change().rolling(window=window).std()
    return df

#linear regression used for future trends:
def predict_trend(df, days=30):
    df = df.dropna()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(df), len(df) + days).reshape(-1, 1)
    future_y = model.predict(future_X)
    return future_X.flatten(), future_y.flatten()

# User inputs:
ticker = "DIS"  #Disney stock
start_date = "2023-01-01"
end_date = "2024-01-01"

df = get_stock_data(ticker, start_date, end_date)
df = compute_moving_averages(df, window=50)
df = compute_volatility(df, window=50)

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Close"], label="Closing Price", color='blue')
plt.plot(df.index, df["MA_50"], label="50-day MA", color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title(f"{ticker} Stock Price & Moving Average")
plt.legend()
plt.show()

future_X, future_y = predict_trend(df)

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Close"], label="Closing Price", color='blue')
plt.plot(pd.date_range(df.index[-1], periods=len(future_X), freq='D'), future_y, label="Predicted Trend", linestyle='dashed', color='green')
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title(f"{ticker} Stock Price Trend Prediction")
plt.legend()
plt.show()