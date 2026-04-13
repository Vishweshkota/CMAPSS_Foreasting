# Turbofan RUL Prediction using CMAPSS

This project predicts Remaining Useful Life (RUL) for turbofan engines using the NASA CMAPSS dataset. It includes end-to-end exploratory data analysis, condition-aware preprocessing, feature engineering, classical regression baselines, and sequence modeling with LSTM.

The workflow was built to compare simple and advanced approaches for prognostics on multivariate engine sensor data across the FD001-FD004 subsets.

## Project Overview

The objective is to estimate how many cycles an engine has left before failure using operational settings and sensor readings.

This project covers:
- exploratory data analysis on engine lifetimes, settings, and sensors
- operating-condition identification using KMeans where needed
- condition-wise sensor normalization
- feature selection and combined dataset creation
- forecasting with Linear Regression and Random Forest
- sequence forecasting with LSTM in PyTorch
- grouped backtesting using `GroupKFold` to avoid engine-level leakage
- official test-set evaluation and model comparison
- saving trained models, outputs, and metadata

## Models Implemented

- Linear Regression
- Random Forest Regressor
- LSTM Regressor using PyTorch

## Key Modeling Design

- Input source: NASA CMAPSS FD001, FD002, FD003, FD004
- Target: Remaining Useful Life (RUL)
- Sequence window size for forecasting: `30`
- Final combined feature count: `22`
- LSTM input shape: `(samples, 30, 22)`
- Backtesting strategy: `GroupKFold` by `unit_id`
- Official test inference: last available window per engine

## Results Summary

### Backtest Results

| Model | MAE | RMSE | R2 |
|------|------:|------:|------:|
| Linear Regression | 34.88 | 47.87 | 0.6190 |
| Random Forest | 29.97 | 44.43 | 0.6718 |
| LSTM | 29.21 | 43.13 | 0.6907 |

### Official Test Results

| Model | MAE | RMSE | R2 |
|------|------:|------:|------:|
| Linear Regression | 27.25 | 34.19 | 0.5518 |
| Random Forest | 22.99 | 31.83 | 0.6115 |
| LSTM | 24.50 | 34.03 | 0.5562 |

Observations:
- LSTM performed best in grouped backtesting.
- Random Forest performed best on the official combined test set in this implementation.
- LSTM showed strong sequence-learning behavior and benefited from shorter optimized training.

## Repository Structure

```text
Project/
  notebooks/
    Turbofan.ipynb
    Turbofan_forecasting.ipynb

  CMAPSSData/
    train_FD001.txt
    train_FD002.txt
    train_FD003.txt
    train_FD004.txt
    test_FD001.txt
    test_FD002.txt
    test_FD003.txt
    test_FD004.txt
    RUL_FD001.txt
    RUL_FD002.txt
    RUL_FD003.txt
    RUL_FD004.txt

  combined_data/
    train_combined.csv
    test_combined.csv
    test_rul_combined.csv
    combined_features.json

  saved_models/
    final_linear_regression.joblib
    final_random_forest.joblib
    final_lstm_optimized.pth
    final_lstm_optimized_metadata.joblib
    final_lstm_optimized_metadata.json
    model_metadata.joblib
    model_metadata.json

  backtest_outputs/
    lr_*.csv / joblib
    rf_*.csv / joblib
    lstm_*.csv / joblib

  final_test_outputs/
    final_model_comparison_with_lstm.csv
    lr_*.csv / joblib
    rf_*.csv / joblib
    lstm_*.csv / joblib

  requirements.txt
  Project_CMAPSS.pdf
