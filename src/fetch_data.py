import os
import yfinance as yf
import pandas as pd

TICKERS = ["SPY", "QQQ", "IWM"]
START_DATE = "2020-01-01"
END_DATE = "2023-01-01"

def fetch_etf_data():
    # Download without auto_adjust to get 'Adj Close'
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False)
    print("Columns in downloaded data:")
    print(data.columns)

    # Extract 'Adj Close' prices
    adj_close = data['Adj Close'].reset_index()
    return adj_close

def combine_data():
    etf_df = fetch_etf_data()

    # Convert from wide to long format
    etf_long = etf_df.melt(id_vars=["Date"], var_name="Ticker", value_name="Adj Close")

    # Ensure data folder exists
    import os
    os.makedirs('data', exist_ok=True)

    # Save the long format CSV
    etf_long.to_csv('data/raw_combined_data.csv', index=False)

    print("Saved raw combined data in long format to data/raw_combined_data.csv")
    print(etf_long.head())


if __name__ == "__main__":
    combine_data()
