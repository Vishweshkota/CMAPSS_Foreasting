# 🚀 Turbofan RUL Prediction using CMAPSS

This project predicts **Remaining Useful Life (RUL)** for turbofan engines using the NASA CMAPSS dataset. It provides an end-to-end pipeline including:

* Exploratory Data Analysis (EDA)
* Condition-aware preprocessing
* Feature engineering
* Classical regression models
* Deep learning with LSTM

The goal is to compare traditional ML models with sequence-based deep learning approaches for predictive maintenance.

---

## 📌 Project Overview

The objective is to estimate **how many operational cycles remain before engine failure**, using:

* Operational settings
* Multivariate sensor readings

### ✔️ Key Components

* Exploratory analysis of engine degradation patterns
* Operating condition identification using **KMeans**
* Condition-wise normalization of sensors
* Feature engineering and dataset merging
* Model training and evaluation
* Backtesting with **GroupKFold** (avoids data leakage)
* Final evaluation on official test sets
* Saving trained models and outputs

---

## 🧠 Models Implemented

* **Linear Regression**
* **Random Forest Regressor**
* **LSTM Regressor (PyTorch)**

---

## ⚙️ Modeling Design

| Feature             | Value                       |
| ------------------- | --------------------------- |
| Dataset             | NASA CMAPSS (FD001–FD004)   |
| Target              | Remaining Useful Life (RUL) |
| Sequence Length     | 30                          |
| Features            | 22                          |
| LSTM Input Shape    | `(samples, 30, 22)`         |
| Validation Strategy | GroupKFold (by `unit_id`)   |
| Test Strategy       | Last window per engine      |

---

## 📊 Results Summary

### 🔁 Backtesting Results

| Model             | MAE       | RMSE      | R²         |
| ----------------- | --------- | --------- | ---------- |
| Linear Regression | 34.88     | 47.87     | 0.6190     |
| Random Forest     | 29.97     | 44.43     | 0.6718     |
| LSTM              | **29.21** | **43.13** | **0.6907** |

---

### 🧪 Official Test Results

| Model             | MAE       | RMSE      | R²         |
| ----------------- | --------- | --------- | ---------- |
| Linear Regression | 27.25     | 34.19     | 0.5518     |
| Random Forest     | **22.99** | **31.83** | **0.6115** |
| LSTM              | 24.50     | 34.03     | 0.5562     |

---

### 🔍 Observations

* LSTM performs best during **grouped backtesting**
* Random Forest generalizes better on the **official test set**
* LSTM captures temporal dependencies effectively

---

## 📁 Repository Structure

```
Project/
│
├── notebooks/
│   ├── Turbofan.ipynb
│   └── Turbofan_forecasting.ipynb
│
├── CMAPSSData/
│   ├── train_*.txt
│   ├── test_*.txt
│   └── RUL_*.txt
│
├── combined_data/
│   ├── train_combined.csv
│   ├── test_combined.csv
│   ├── test_rul_combined.csv
│   └── combined_features.json
│
├── saved_models/
│   ├── final_linear_regression.joblib
│   ├── final_lstm_optimized.pth
│   ├── final_lstm_optimized_metadata.*
│   └── model_metadata.*
│
├── backtest_outputs/
├── final_test_outputs/
│
├── requirements.txt
└── Project_CMAPSS.pdf
```

---

## ⚠️ Model Files

The trained model **`final_random_forest.joblib` (~1 GB)** is **not included** in this repository due to GitHub file size limitations.

### 📥 Download Model

👉 **[Add your model download link here]**

### 📂 Placement

After downloading, place it here:

```
saved_models/final_random_forest.joblib
```

### ▶️ Usage

Ensure the file is present before running inference or evaluation scripts.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

1. Open notebooks:

   * `Turbofan.ipynb`
   * `Turbofan_forecasting.ipynb`

2. Run cells sequentially for:

   * preprocessing
   * training
   * evaluation

---

## 📚 Dataset

* NASA CMAPSS Turbofan Engine Dataset
* Includes FD001–FD004 subsets with varying operating conditions and fault modes

---

## 📌 Key Highlights

* Handles **multi-condition environments**
* Avoids leakage using **GroupKFold**
* Combines classical ML + deep learning
* Structured and reproducible pipeline

---

## 📎 License

This project is for academic and research purposes.
