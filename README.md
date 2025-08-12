# ETF Swing Trading Project

## Problem
Swing trading ETFs like SPY, QQQ, and IWM needs accurate short-term predictions to improve profits. Simple buy-and-hold often misses chances to gain more or avoid losses during market swings.

## Solution
This project uses historical ETF prices and economic data to create features like moving averages and RSI. It trains machine learning models to predict if prices will rise in the next 5 days. Based on these predictions, it runs a trading simulation that only buys when the model expects gains. The results are compared to basic buy-and-hold to show better performance.

## Data Sources
- **YFinance API:** Adjusted close prices for ETFs SPY, QQQ, and IWM from 2015 to 2024.  
- **FRED Economic Data:** US 10-Year Treasury Constant Maturity Rate as an additional macroeconomic indicator.

## Data Processing & Features
- Combined both data sources into a unified dataset.  
- Generated 20+ features including moving averages (SMA, EMA), RSI, volatility, and Treasury rate binned dummy variables.  
- Created the target label as binary (price increase over next 5 days).

## Folder Structure
<img width="752" height="460" alt="image" src="https://github.com/user-attachments/assets/2dd0a13f-2a2b-473a-8866-4b5e7eba57cf" />

## Modeling
- Trained two models:  
  - **Random Forest Classifier** (classic model from lectures)  
  - **LightGBM Classifier** (state-of-the-art gradient boosting)  
- Used time series cross-validation for model evaluation.  
- Models output binary predictions on price movement.

## Trading Simulation
- Simulated a strategy holding ETFs only when model predicts price increase.  
- Calculated key performance metrics:  
  - CAGR (Compound Annual Growth Rate)  
  - Sharpe Ratio  
  - Max Drawdown  
- Compared strategy returns against simple buy-and-hold benchmark.

## Automation
- All steps automated via `scripts/run_all.sh`.  
- Dependencies managed with `requirements.txt` and virtual environment.  
- Easy to rerun full pipeline end-to-end.

## How to Run
```bash
git clone https://github.com/glk08909/etf-swing-trading-multisource.git
cd etf-swing-trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/run_all.sh
./scripts/run_all.sh

