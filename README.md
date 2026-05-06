# DSToolkit.jl 🚀

A high-level, unified Machine Learning toolkit for Julia. DSToolkit abstracts away `MLJ.jl`, `Flux.jl`, and `StateSpaceModels.jl` behind a single, consistent API for **Tabular Regression**, **Tabular Classification**, **Statistical Time Series Forecasting**, and **Deep Learning Time Series**.

## Features

- **Flexible Data Ingestion** — Accepts `DataFrame`, `Matrix`, `CSV`, `JLD2`, `Dict`, `NamedTuple`, or raw `Vector`
- **Auto Task Detection** — Reads your target data and detects regression vs classification vs time series
- **22 Algorithms** — All popular ML methods in one toolkit
- **Auto-Compare** — Train all suitable models, compare metrics, display results, pick the best
- **Robust Persistence** — Save trained models to disk and load them back for inference

## Supported Models

### Tabular Regression (9)
| Model | Backend Package |
|---|---|
| Linear Regression | `GLM` |
| Ridge Regression | `MLJLinearModels` |
| Lasso Regression | `MLJLinearModels` |
| Elastic Net | `MLJLinearModels` |
| Decision Tree | `DecisionTree` |
| Random Forest | `DecisionTree` |
| XGBoost | `XGBoost` |
| KNN | `NearestNeighborModels` |
| SVM (Epsilon SVR) | `LIBSVM` |

### Tabular Classification (8)
| Model | Backend Package |
|---|---|
| Logistic Regression | `MLJLinearModels` |
| Decision Tree | `DecisionTree` |
| Random Forest | `DecisionTree` |
| AdaBoost Stump | `DecisionTree` |
| XGBoost | `XGBoost` |
| KNN | `NearestNeighborModels` |
| SVM | `LIBSVM` |
| Gaussian Naive Bayes | `NaiveBayes` |

### Time Series — Statistical (2)
| Model | Backend Package |
|---|---|
| ARIMA | `StateSpaceModels` |
| ETS (Exponential Smoothing) | `StateSpaceModels` |

### Time Series — Deep Learning (3)
| Model | Backend Package |
|---|---|
| RNN | `Flux` |
| LSTM | `Flux` |
| GRU | `Flux` |

## Installation

Since this is an unregistered local package, open the Julia REPL, press `]` to enter the Pkg prompt, and run:

```julia
pkg> develop C:/path/to/DSToolkitJulia
```

Or from a script:

```julia
using Pkg
Pkg.develop(path="C:/path/to/DSToolkitJulia")
```

## Quick Start

### 1. Ingest Data (Flexible Input)

```julia
using DSToolkit

# From a DataFrame (auto-detects task type)
data = ingest_data(df; target=:price)

# From a CSV file
data = ingest_data("housing_data.csv"; target=:price)

# From a Matrix + Vector
data = ingest_data(X_matrix, y_vector)

# From a Dict (e.g., output from a preprocessing pipeline)
data = ingest_data(Dict("X" => features, "y" => labels))

# Univariate time series
data = ingest_data(sales_vector; task=:timeseries)

# Override auto-detection
data = ingest_data(df; target=:class, task=:classification)
```

### 2. Auto-Compare All Models

```julia
result = auto_compare(data)
# Trains all 9 regression (or 8 classification) models,
# prints a comparison table, and returns the best model.

# Access results:
result.best_model       # The top-performing model
result.best_model_name  # e.g., "XGBoost"
result.results          # DataFrame of all metrics
result.all_models       # Vector of (name => model) pairs
```

### 3. Train a Specific Model

```julia
X_train, X_test, y_train, y_test = train_test_split(data)

model = XGBoostReg(num_round=200, max_depth=8)
fit!(model, X_train, y_train)

preds = predict(model, X_test)
metrics = evaluate(model, X_test, y_test)
# Dict("RMSE" => 0.032, "MAE" => 0.025, "R²" => 0.987, "MAPE" => 3.2)
```

### 4. Save & Load Models

```julia
# Save
save_model(result.best_model, "my_best_model")

# Load (in another session or script)
model = load_toolkit_model("my_best_model.jld2")

# Use for inference on new data
predictions = predict(model, new_data)
```

### 5. Time Series Workflow

```julia
# Statistical (ARIMA / ETS)
data = ingest_data(y_vector; task=:timeseries)
result = auto_compare(data)
forecast = predict(result.best_model, 30)  # 30 steps ahead

# Deep Learning (RNN / LSTM / GRU)
X_seq = rand(Float32, n_features, seq_len, n_samples)
y_seq = rand(Float32, 1, n_samples)
X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y_seq)
result = auto_compare(X_tr, y_tr, X_te, y_te)
```

## Evaluation Metrics

| Task | Metrics |
|---|---|
| Regression | RMSE, MAE, R², MAPE |
| Classification | Accuracy, Macro-Precision, Macro-Recall, Macro-F1 |
| Time Series | RMSE, MAE, MAPE |

## Pipeline Integration

DSToolkit is designed as **Step 2** in a data pipeline:

```
[Step 1: Data Preprocessing] → cleaned data (any format)
        ↓
[Step 2: DSToolkit] → ingest_data() → auto_compare() or fit!()
        ↓
[Step 3: Deployment] → save_model() / load_toolkit_model() → predict()
```

The `ingest_data()` function accepts whatever format your preprocessing step outputs — `DataFrame`, `Matrix`, `Dict`, `NamedTuple`, CSV file, or JLD2 file. As long as the data is clean, DSToolkit handles the rest.

## License

MIT