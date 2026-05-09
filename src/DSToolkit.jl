module DSToolkit

# Dependencies
using DataFrames, Statistics, Random
using CategoricalArrays
using PrettyTables
using JLD2, CSV
import MLJ
import Flux
import StateSpaceModels
const SSM = StateSpaceModels

# MLJ interface packages (required for MLJ.@load to work)
import MLJGLMInterface
import MLJDecisionTreeInterface
import MLJLIBSVMInterface
import MLJNaiveBayesInterface
import MLJXGBoostInterface
import MLJLinearModels
import NearestNeighborModels

# Sub-modules (order matters — types first, then logic)
include("config.jl")

# Type definitions (split into 4 focused files)
include("types/base.jl")
include("types/flux_layers.jl")
include("types/models.jl")
include("types/containers.jl")

# Data utilities
include("data/validation.jl")
include("ingest.jl")
include("split.jl")
include("data/loader.jl")
include("data/preprocessing.jl")

# Training and prediction
include("training/fit.jl")
include("predict.jl")
include("evaluation/evaluate.jl")
include("comparison/comparison.jl")
include("persistence.jl")

# Inference
include("inference/predictor.jl")
include("inference/batch.jl")

# Visualization
include("visualization/distributions.jl")
include("visualization/correlations.jl")
include("visualization/model_comparison.jl")
include("visualization/timeseries.jl")

# Exports — Public API

# Data ingestion & splitting
export ingest_data, detect_task, train_test_split, ToolkitData

# Dataset loaders
export load_iris, load_titanic, load_wine_quality
export load_housing, load_diabetes, load_synthetic_reg
export load_airline_passengers, load_stock_prices, load_temperature
export list_datasets

# Preprocessing
export impute_missing, standardize, normalize
export one_hot_encode, label_encode
export add_polynomial_features, add_interaction_features

# Core workflow
export fit!, predict, predict_proba, evaluate, auto_compare
export make_timeseries_windows
export TRAINING_CONFIG, training_config

# Comparison result
export ComparisonResult

# Model persistence
export save_model, load_toolkit_model

# Inference
export inference, inference_with_preprocessing
export batch_predict, parallel_predict

# Visualization
export plot_histogram, plot_boxplot, plot_target_distribution
export plot_correlation_heatmap, plot_feature_vs_target
export plot_comparison_results, plot_metric_comparison
export plot_timeseries, plot_forecast

# --- Regression models ---
export LinearReg, RidgeReg, LassoReg, ElasticNetReg
export DecisionTreeReg, RandomForestReg, XGBoostReg, KNNReg, SVMReg

# --- Classification models ---
export LogisticCls, DecisionTreeCls, RandomForestCls, AdaBoostCls
export XGBoostCls, KNNCls, SVMCls, NaiveBayesCls

# --- Time series models ---
export ARIMAModel, ETSModel
export RNNModel, LSTMModel, GRUModel

# --- Type hierarchy (for extension) ---
export AbstractToolkitModel, TabularModel, RegressionModel, ClassificationModel
export TimeSeriesModel, DeepTimeSeriesModel, StatTimeSeriesModel

end # module DSToolkit
