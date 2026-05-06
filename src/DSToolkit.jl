module DSToolkit

# ==============================================================================
# Dependencies
# ==============================================================================
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

# ==============================================================================
# Sub-modules (order matters — types first, then logic)
# ==============================================================================
include("types.jl")
include("ingest.jl")
include("split.jl")
include("fit.jl")
include("predict.jl")
include("evaluate.jl")
include("compare.jl")
include("persistence.jl")

# ==============================================================================
# Exports — Public API
# ==============================================================================

# Data ingestion & splitting
export ingest_data, detect_task, train_test_split, ToolkitData

# Core workflow
export fit!, predict, predict_proba, evaluate, auto_compare

# Comparison result
export ComparisonResult

# Model persistence
export save_model, load_toolkit_model

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