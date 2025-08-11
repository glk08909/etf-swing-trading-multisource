#!/bin/bash

# Run fetch data
python3 src/fetch_data.py

# Build features
python3 src/build_features.py

# Train models
python3 src/train_model.py

# Backtest
python3 src/backtest_strategy.py
