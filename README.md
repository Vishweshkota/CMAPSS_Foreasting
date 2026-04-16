# Turbofan RUL Prediction Using CMAPSS

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20API-black)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An end-to-end predictive maintenance project for estimating **Remaining Useful Life (RUL)** of turbofan engines using the NASA **CMAPSS** dataset. This repository combines exploratory analysis, condition-aware preprocessing, feature engineering, classical machine learning, sequence-based deep learning with **LSTM**, and a local **Flask** web application for inference, visualization, and API testing.

The project compares traditional regression models with sequence models and deploys the final inference workflow through a browser UI, REST API, Swagger documentation, and Postman collection.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Modeling Design](#modeling-design)
- [Results Summary](#results-summary)
- [Deployment Features](#deployment-features)
- [Local Web Application](#local-web-application)
- [API Endpoints](#api-endpoints)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Input Requirements](#input-requirements)
- [Example Workflow](#example-workflow)
- [Tech Stack](#tech-stack)
- [Postman and Swagger](#postman-and-swagger)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

---

## Project Overview

The objective of this project is to predict **how many operational cycles remain before an engine fails** using:

- operational settings
- multivariate sensor readings
- engine degradation history across time

This repository covers the full workflow from raw CMAPSS data to deployed local inference.

The pipeline includes:

- exploratory analysis of engine degradation behavior
- operating-condition identification using **KMeans**
- condition-wise normalization of sensors
- feature engineering and combined dataset creation
- model training and evaluation
- grouped backtesting with **GroupKFold** to avoid leakage
- final evaluation on official CMAPSS test sets
- saving trained models and preprocessing artifacts
- local deployment through a Flask API and browser UI
- API documentation with **Swagger**
- API testing with **Postman**

---

## Objectives

- build a reliable **RUL prediction pipeline** for turbofan engines
- compare classical machine learning and deep learning approaches
- avoid data leakage using grouped validation by engine
- support both notebook experimentation and deployable inference
- provide a practical local interface for testing predictions

---

## Key Features

### Machine Learning Pipeline

- combined use of all CMAPSS subsets: `FD001`, `FD002`, `FD003`, `FD004`
- operating-condition clustering with **KMeans**
- condition-specific preprocessing and normalization
- engineered and selected features for model input
- grouped cross-validation with **GroupKFold**
- final testing on official CMAPSS test sets

### Model Comparison

- **Linear Regression**
- **Random Forest Regressor**
- **LSTM Regressor (PyTorch)**

### Deployment and Inference

- single-row streaming prediction
- compare-all-models streaming prediction
- full-window prediction
- CSV file upload prediction
- rolling 30-cycle engine state management
- reset specific engine state
- reset all in-memory states
- browser-based UI
- Swagger API docs
- Postman collection for endpoint testing

---

## Dataset

This project uses the **NASA CMAPSS turbofan engine degradation dataset**.

### CMAPSS subsets used

- `FD001`
- `FD002`
- `FD003`
- `FD004`

### Prediction target

- **Remaining Useful Life (RUL)**

### Raw input signals

- 3 operational settings
- 21 sensor measurements
- cycle-by-cycle engine trajectories

---

## Models Implemented

### 1. Linear Regression

A simple baseline model trained on flattened 30-cycle windows.

### 2. Random Forest Regressor

A non-linear ensemble model trained on flattened 30-cycle windows.

### 3. LSTM Regressor

A sequence model implemented in **PyTorch** that learns temporal degradation patterns directly from 30-step sequences.

---

## Modeling Design

| Feature | Value |
|---|---|
| Dataset | NASA CMAPSS (`FD001` to `FD004`) |
| Target | Remaining Useful Life (RUL) |
| Sequence Length | `30` |
| Final Feature Count | `22` |
| LSTM Input Shape | `(samples, 30, 22)` |
| LR/RF Input Shape | `(samples, 660)` |
| Validation Strategy | `GroupKFold` by `unit_id` |
| Test Strategy | last window per engine |

### Design Notes

- each training sample is created from a **30-cycle sliding window**
- **GroupKFold** is used to prevent engine-level leakage
- the **LSTM** receives sequence input directly
- **Linear Regression** and **Random Forest** use flattened windows
- final test evaluation uses the **last valid window per engine**

---

## Results Summary

### Backtesting Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 34.88 | 47.87 | 0.6190 |
| Random Forest | 29.97 | 44.43 | 0.6718 |
| LSTM | **29.21** | **43.13** | **0.6907** |

### Official Test Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 27.25 | 34.19 | 0.5518 |
| Random Forest | **22.99** | **31.83** | **0.6115** |
| LSTM | 24.50 | 34.03 | 0.5562 |

### Observations

- **LSTM performed best during grouped backtesting**
- **Random Forest generalized best on the official combined test set**
- **LSTM captured temporal degradation patterns effectively**
- **Grouped validation was essential to avoid leakage from sliding windows**

---

## Deployment Features

This project goes beyond notebooks and includes a full local deployment workflow for interactive prediction.

### Web UI Features

- single-row streaming prediction
- compare-all-models prediction
- rolling 30-cycle progress tracking
- prediction history table
- model comparison chart
- CSV upload prediction
- reset current engine state
- reset all engine states
- raw JSON response viewer
- direct access to Swagger API docs

### File Upload Prediction

Users can upload a CSV file containing engine sensor readings and request a prediction using the last `N` rows.

The file upload pipeline:

- accepts CSV input from the UI or API
- filters by selected `engine_id` if present
- validates required columns
- preprocesses each row using saved preprocessing artifacts
- builds a fixed 30-row model window
- pads or truncates rows when necessary
- returns a final predicted RUL

### Compare-All-Models Mode

The backend supports predicting with all three models on the same shared streaming update, allowing fast comparison of:

- Linear Regression
- Random Forest
- LSTM

---

## Local Web Application

A local **Flask** backend is provided for interactive inference and testing.

### Backend Capabilities

- loads trained ML and LSTM models once at startup
- loads preprocessing artifacts once at startup
- preprocesses raw streaming rows
- maintains rolling engine history in memory
- supports prediction from:
  - one raw row at a time
  - a full processed 30-row window
  - an uploaded CSV file
- returns prediction results in JSON format
- exposes interactive Swagger docs through **Flasgger**

### Browser UI

After starting the app, open:

```text
http://127.0.0.1:5000/
```

### Swagger API Docs

```text
http://127.0.0.1:5000/apidocs/
```

### Health Check

```text
http://127.0.0.1:5000/health
```

---

## API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Browser UI |
| `/health` | GET | Backend health check |
| `/models` | GET | Model and preprocessing metadata |
| `/predict/window` | POST | Predict from a full processed 30-row window |
| `/predict/stream` | POST | Predict from one raw streaming sensor row |
| `/predict/stream/all` | POST | Predict with all three models from one shared stream update |
| `/predict/file` | POST | Predict from uploaded CSV file |
| `/engines/reset` | POST | Reset one engine state |
| `/engines/reset_all` | POST | Reset all engine states |
| `/apidocs/` | GET | Swagger API documentation |

---

## Repository Structure

```text
CMAPSS_Foreasting/
│
├── backend/
│   ├── app.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   ├── state.py
│   ├── inference.py
│   └── templates/
│       └── index.html
│
├── notebooks/
│   ├── Turbofan.ipynb
│   └── Turbofan_forecasting.ipynb
│
├── CMAPSSData/
│   ├── train_FD001.txt
│   ├── train_FD002.txt
│   ├── train_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD001.txt
│   ├── test_FD002.txt
│   ├── test_FD003.txt
│   ├── test_FD004.txt
│   ├── RUL_FD001.txt
│   ├── RUL_FD002.txt
│   ├── RUL_FD003.txt
│   └── RUL_FD004.txt
│
├── combined_data/
│   ├── train_combined.csv
│   ├── test_combined.csv
│   ├── test_rul_combined.csv
│   ├── combined_features.json
│   └── preprocessing_summary.json
│
├── saved_models/
│   ├── final_linear_regression.joblib
│   ├── final_random_forest.joblib
│   ├── final_lstm_optimized.pth
│   ├── final_lstm_optimized_metadata.joblib
│   ├── final_lstm_optimized_metadata.json
│   ├── model_metadata.joblib
│   ├── model_metadata.json
│   └── preprocessing_artifacts.joblib
│
├── backtest_outputs/
├── final_test_outputs/
│
├── Postman/
│   └── JetEngineRulPrediction.postman_collection.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Prerequisites

Make sure you have the following installed:

- Python `3.10+`
- `pip`
- virtual environment support

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd CMAPSS_Foreasting
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Run from the `backend` Folder

```bash
cd backend
python app.py
```

### Option 2: Run from the Project Root

```bash
python backend/app.py
```

### Open the Application

- Browser UI: `http://127.0.0.1:5000/`
- Swagger Docs: `http://127.0.0.1:5000/apidocs/`
- Health Check: `http://127.0.0.1:5000/health`

---

## Input Requirements

### Required Raw Fields for Streaming Prediction

A streaming row should include:

- `tag`
- `engine_id`
- `time_cycles`
- `setting_1`
- `setting_2`
- `setting_3`
- `sensor_1`
- `sensor_2`
- `sensor_3`
- `sensor_4`
- `sensor_5`
- `sensor_6`
- `sensor_7`
- `sensor_8`
- `sensor_9`
- `sensor_10`
- `sensor_11`
- `sensor_12`
- `sensor_13`
- `sensor_14`
- `sensor_15`
- `sensor_16`
- `sensor_17`
- `sensor_18`
- `sensor_19`
- `sensor_20`
- `sensor_21`

### Supported Dataset Tags

- `FD001`
- `FD002`
- `FD003`
- `FD004`

### Supported Model Names

- `lr`
- `rf`
- `lstm`

### CSV Upload Requirements

For file-based prediction, the uploaded CSV should include:

- `time_cycles`
- `setting_1`
- `setting_2`
- `setting_3`
- `sensor_1` to `sensor_21`

Optional:

- `engine_id`

If multiple engines are present in the CSV, `engine_id` is used to filter the selected engine before prediction.

---

## Example Workflow

### 1. Start the Backend

```bash
cd backend
python app.py
```

### 2. Open the Browser UI

```text
http://127.0.0.1:5000/
```

### 3. Choose a Prediction Mode

- stream one row at a time
- compare predictions from all three models
- upload a CSV file and predict from the last `N` rows

### 4. Test the API

Use either:

- Swagger: `http://127.0.0.1:5000/apidocs/`
- Postman collection: `Postman/JetEngineRulPrediction.postman_collection.json`

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **PyTorch**
- **Flask**
- **Flasgger**
- **Matplotlib**
- **Seaborn**
- **JupyterLab**
- **Postman**

---

## Postman and Swagger

### Swagger

Interactive API documentation is available at:

```text
http://127.0.0.1:5000/apidocs/
```

### Postman

A Postman collection is included for testing the API:

```text
Postman/JetEngineRulPrediction.postman_collection.json
```

This collection can be imported directly into Postman for local testing of:

- health check
- model metadata
- streaming prediction
- compare-all-models prediction
- file upload prediction
- engine reset endpoints

---

## Future Improvements

- deploy the API to cloud infrastructure
- add batch inference for multiple engines
- include model explainability for predictions
- improve uncertainty estimation
- add production-ready configuration and authentication
- enhance the frontend with richer visual analytics
- package the app with Docker for easier deployment

---

## Conclusion

This project demonstrates a complete **predictive maintenance workflow** for turbofan engines using the NASA CMAPSS dataset, from raw data analysis and model training to deployment through a local Flask web application.

It highlights the tradeoff between classical machine learning and sequence modeling:

- **LSTM** performed best during grouped backtesting
- **Random Forest** generalized best on the official combined test set

The final result is a practical, end-to-end system for **Remaining Useful Life prediction** with research, evaluation, deployment, and API testing all included in one repository.
