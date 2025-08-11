import pandas as pd
import numpy as np

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def build_features():
    df = pd.read_csv("data/raw_combined_data.csv", parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # Technical indicators
    df["SMA_10"] = df.groupby("Ticker")["Adj Close"].transform(lambda x: x.rolling(10).mean())
    df["SMA_50"] = df.groupby("Ticker")["Adj Close"].transform(lambda x: x.rolling(50).mean())
    df["EMA_20"] = df.groupby("Ticker")["Adj Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["RSI_14"] = df.groupby("Ticker")["Adj Close"].transform(rsi)

    # Volatility (rolling std dev) - uncomment if needed
    # df["Volatility_10"] = df.groupby("Ticker")["Adj Close"].transform(lambda x: x.pct_change().rolling(10).std())

    # Comment out treasury part if column missing
    # df["Treasury_10Y_Bin"] = pd.cut(df["Treasury_10Y"], bins=5, labels=False)
    # treasury_dummies = pd.get_dummies(df["Treasury_10Y_Bin"], prefix="T10Y")
    # df = pd.concat([df, treasury_dummies], axis=1)

    # Future return (5 days ahead)
    df["Future_Close"] = df.groupby("Ticker")["Adj Close"].shift(-5)
    df["Future_Return"] = (df["Future_Close"] - df["Adj Close"]) / df["Adj Close"]
    df["Target"] = (df["Future_Return"] > 0).astype(int)

    df.dropna(inplace=True)

    # Save features
    df.to_csv("data/features.csv", index=False)
    print(df.columns)
    print("Feature dataset saved to data/features.csv")

if __name__ == "__main__":
    build_features()
