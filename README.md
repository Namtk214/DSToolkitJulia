# DSToolkit.jl

A high-level, unified Machine Learning toolkit for Julia. DSToolkit wraps `MLJ.jl`, `Flux.jl`, and `StateSpaceModels.jl` behind a single consistent API for **Tabular Regression**, **Tabular Classification**, and **Time Series Forecasting** (statistical and deep learning).

## Features

- **9 Built-in Datasets** — ready-to-use datasets for classification, regression, and time series
- **Flexible Data Ingestion** — accepts `DataFrame`, `Matrix`, `CSV`, `Dict`, `NamedTuple`, or raw `Vector`
- **Data Preprocessing** — missing value imputation, feature scaling, encoding, and feature engineering
- **Auto Task Detection** — infers regression / classification / timeseries from target type
- **22 Algorithms** — all major ML methods behind one interface
- **Auto-Compare** — trains all suitable models, ranks by metric, returns the best
- **Visualization** — distribution plots, correlation heatmaps, model comparison charts, time series plots
- **Inference API** — batch and parallel prediction with optional preprocessing pipelines
- **Model Persistence** — save trained models to disk and reload in any session
- **Web Demo** — interactive HTTP demo server for exploring datasets and running predictions

## Supported Models

### Tabular Regression (9)
| Model | Backend |
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
| Model | Backend |
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
| Model | Backend |
|---|---|
| ARIMA | `StateSpaceModels` |
| ETS (Exponential Smoothing) | `StateSpaceModels` |

### Time Series — Deep Learning (3)
| Model | Backend |
|---|---|
| RNN | `Flux` |
| LSTM | `Flux` |
| GRU | `Flux` |

## Installation

This is an unregistered local package. From the Julia REPL, press `]` to enter the Pkg prompt:

```julia
pkg> develop /path/to/DSToolkitJulia
```

Or from a script:

```julia
using Pkg
Pkg.develop(path="/path/to/DSToolkitJulia")
```

## Quick Start

### 1. Ingest Data

```julia
using DSToolkit

# From a DataFrame — auto-detects task type from target column
data = ingest_data(df; target=:price)

# From a CSV file
data = ingest_data("housing_data.csv"; target=:price)

# From a Matrix + Vector
data = ingest_data(X_matrix, y_vector)

# Univariate time series
data = ingest_data(sales_vector; task=:timeseries)

# Override auto-detection
data = ingest_data(df; target=:class, task=:classification)
```

### 2. Auto-Compare All Models

```julia
result = auto_compare(data)
# Trains all suitable models, prints a ranked comparison table,
# and returns the best model.

result.best_model       # top-performing trained model
result.best_model_name  # e.g. "XGBoost"
result.results          # DataFrame of all metrics
result.all_models       # Vector of (name => model) pairs
```

### 3. Train a Specific Model

```julia
X_train, X_test, y_train, y_test = train_test_split(data)

model = XGBoostReg(num_round=200, max_depth=8)
fit!(model, X_train, y_train)

preds   = predict(model, X_test)
metrics = evaluate(model, X_test, y_test)
# Dict("RMSE" => 0.032, "MAE" => 0.025, "R²" => 0.987, "MAPE" => 3.2)
```

### 4. Save & Load Models

```julia
save_model(result.best_model, "my_best_model")

model = load_toolkit_model("my_best_model.jld2")
predictions = predict(model, new_data)
```

### 5. Time Series Workflow

```julia
# Statistical (ARIMA / ETS)
data   = ingest_data(y_vector; task=:timeseries)
result = auto_compare(data)
forecast = predict(result.best_model, 30)   # 30 steps ahead

# Deep Learning (RNN / LSTM / GRU)
X_seq = rand(Float32, n_features, seq_len, n_samples)
y_seq = rand(Float32, 1, n_samples)
X_tr, X_te, y_tr, y_te = train_test_split(X_seq, y_seq)
result = auto_compare(X_tr, y_tr, X_te, y_te)
```

## Built-in Datasets

### Classification
```julia
data = load_iris()          # 150 samples, 4 features, 3 classes
data = load_titanic()       # 891 samples, binary classification
data = load_wine_quality()  # 1 599 samples, quality ratings 3–8
```

### Regression
```julia
data = load_housing()       # 506 samples, 13 features (Boston Housing)
data = load_diabetes()      # 442 samples, 10 clinical features
data = load_synthetic_reg() # 1 000 samples, clean synthetic data
```

### Time Series
```julia
data = load_airline_passengers() # 144 months, seasonal trend
data = load_stock_prices()       # 252 days, financial series
data = load_temperature()        # 365 days, seasonal weather
```

```julia
# List all datasets
list_datasets()
# => Dict(:classification => [...], :regression => [...], :timeseries => [...])
```

## Preprocessing

```julia
# Missing value imputation — strategies: :mean, :median, :mode, :forward
data = impute_missing(data; strategy=:mean)

# Feature scaling
data = standardize(data)   # z-score  (mean=0, std=1)
data = normalize(data)     # min-max  [0, 1]

# Categorical encoding
data = one_hot_encode(data, [:color, :size])
data = label_encode(data,   [:category])

# Feature engineering
data = add_polynomial_features(data; degree=2)
data = add_interaction_features(data, [(:feature1, :feature2)])

# Chaining steps
data = load_housing()
data = impute_missing(data; strategy=:mean)
data = standardize(data)
result = auto_compare(data)
```

## Visualization

Requires `Plots.jl` and `StatsPlots.jl`.

```julia
using Plots

plot_histogram(data, :feature_name)
plot_boxplot(data, :feature_name)
plot_target_distribution(data)

plot_correlation_heatmap(data)
plot_feature_vs_target(data, :feature_name)

result = auto_compare(data)
plot_comparison_results(result)

plot_timeseries(y_train, y_test, predictions)
plot_forecast(model, y_train, horizon=30)
```

## Inference API

```julia
# Single prediction
predictions = inference(model, new_data)

# With preprocessing
preprocess_fn = x -> standardize(impute_missing(x))
predictions = inference_with_preprocessing(model, raw_data, preprocess_fn)

# Batch and parallel prediction
results = batch_predict(model, [batch1, batch2, batch3])
results = parallel_predict(model, [chunk1, chunk2, chunk3])
```

## Evaluation Metrics

| Task | Metrics |
|---|---|
| Regression | RMSE, MAE, R², MAPE |
| Classification | Accuracy, Macro-Precision, Macro-Recall, Macro-F1 |
| Time Series | RMSE, MAE, MAPE |

## Demo Server

An interactive web demo is included. It runs on plain `HTTP.jl` with no additional dependencies.

```julia
julia --project=. demo/demo_server.jl
# Server starts at http://localhost:8000
```

Pages:
| URL | Description |
|---|---|
| `/` | Home — select dataset and model, train, view metrics |
| `/inference` | Run live predictions against the trained model |
| `/visualization` | Explore dataset distributions, correlations, and sample rows |

## Project Structure

```
DSToolkitJulia/
├── src/
│   ├── DSToolkit.jl        # module entry point and exports
│   ├── config.jl           # centralised training defaults
│   ├── ingest.jl           # data ingestion
│   ├── split.jl            # train/test splitting
│   ├── predict.jl          # predict() interface
│   ├── persistence.jl      # save / load models
│   ├── types/              # abstract and concrete model types
│   ├── data/               # loaders and preprocessing
│   ├── models/             # regression, classification, timeseries
│   ├── training/           # fit! implementations
│   ├── evaluation/         # metrics
│   ├── comparison/         # auto_compare and display
│   ├── inference/          # batch and parallel prediction
│   └── visualization/      # plotting utilities
├── demo/
│   ├── demo_server.jl      # HTTP demo server
│   └── views/              # HTML templates
├── test/
│   └── runtests.jl         # test suite (50+ test cases)
├── data/                   # built-in CSV datasets
└── Project.toml
```

## License

MIT
