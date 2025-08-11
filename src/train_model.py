import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import joblib

import os
os.makedirs("models", exist_ok=True)


def train_models():
    df = pd.read_csv("data/features.csv", parse_dates=["Date"])
    
    # Features: all except non-numeric & target columns
    exclude_cols = ["Date", "Ticker", "Future_Close", "Future_Return", "Target"]
    features = [col for col in df.columns if col not in exclude_cols]

    X = df[features]
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=5)

    # Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        rf_model.fit(X_train, y_train)
        preds = rf_model.predict(X_test)
        rf_scores.append(accuracy_score(y_test, preds))
    print("Random Forest CV accuracies:", rf_scores)
    joblib.dump(rf_model, "models/rf_model.pkl")

    # LightGBM model
    lgb_model = lgb.LGBMClassifier()
    lgb_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        lgb_model.fit(X_train, y_train)
        preds = lgb_model.predict(X_test)
        lgb_scores.append(accuracy_score(y_test, preds))
    print("LightGBM CV accuracies:", lgb_scores)
    joblib.dump(lgb_model, "models/lgb_model.pkl")

if __name__ == "__main__":
    train_models()
