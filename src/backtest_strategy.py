import os
import pandas as pd
import joblib
import numpy as np

def sharpe_ratio(returns, risk_free=0.0):
    excess_returns = returns - risk_free / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def backtest():
    df = pd.read_csv("data/features.csv", parse_dates=["Date"])
    
    # Normalize column names
    #df.columns = df.columns.str.replace(" ", "_")
    
    
    # Check model files exist
    for model_file in ["models/rf_model.pkl", "models/lgb_model.pkl"]:
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found.")

    models = {
        "rf": joblib.load("models/rf_model.pkl"),
        "lgb": joblib.load("models/lgb_model.pkl")
    }

    exclude_cols = {"Date", "Ticker", "Future_Close", "Future_Return", "Target"}    
    features = [col for col in df.columns if col not in ["Date", "Ticker", "Future_Close", "Future_Return", "Target"]]    

    results = {}
    for model_name, model in models.items():
        df[f"Pred_{model_name}"] = model.predict(df[features])
        df[f"Position_{model_name}"] = df[f"Pred_{model_name}"]

        df[f"Daily_Return_{model_name}"] = df.groupby("Ticker")["Adj Close"].pct_change() * df[f"Position_{model_name}"]        
        df[f"Equity_Curve_{model_name}"] = (1 + df[f"Daily_Return_{model_name}"]).groupby(df["Ticker"]).cumprod()

        metrics = {}
        for ticker in df["Ticker"].unique():
            ticker_df = df[df["Ticker"] == ticker].copy()
            eq = ticker_df[f"Equity_Curve_{model_name}"]
            returns = ticker_df[f"Daily_Return_{model_name}"].dropna()
            if len(eq) < 2 or returns.std() == 0:
                # Skip tickers with too few data or zero volatility
                continue

            metrics[ticker] = {
                "CAGR": (eq.iloc[-1]) ** (252 / len(eq)) - 1,
                "Sharpe": sharpe_ratio(returns),
                "Max_Drawdown": max_drawdown(eq)
            }
        results[model_name] = metrics

    for model_name, metrics in results.items():
        print(f"Performance metrics for {model_name}:")
        for ticker, vals in metrics.items():
            print(f"  {ticker}: CAGR={vals['CAGR']:.2%}, Sharpe={vals['Sharpe']:.2f}, Max Drawdown={vals['Max_Drawdown']:.2%}")
        print()

    df.to_csv("data/backtest_results.csv", index=False)
    print("Backtest results saved to data/backtest_results.csv")

if __name__ == "__main__":
    backtest()
