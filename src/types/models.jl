# Model Type Definitions for DSToolkit
# All 22 concrete model types (9 regression, 8 classification, 5 time series)

# ============================================================================
# REGRESSION MODELS (9)
# ============================================================================

mutable struct LinearReg <: RegressionModel
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    LinearReg() = new(false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct RidgeReg <: RegressionModel
    lambda::Float64
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    RidgeReg(; lambda=training_config(:model_defaults, :ridge_reg).lambda) =
        new(lambda, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct LassoReg <: RegressionModel
    lambda::Float64
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    LassoReg(; lambda=training_config(:model_defaults, :lasso_reg).lambda) =
        new(lambda, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct ElasticNetReg <: RegressionModel
    lambda::Float64
    alpha::Float64
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    ElasticNetReg(; lambda=training_config(:model_defaults, :elastic_net_reg).lambda,
                  alpha=training_config(:model_defaults, :elastic_net_reg).alpha) =
        new(lambda, alpha, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct DecisionTreeReg <: RegressionModel
    max_depth::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    DecisionTreeReg(; max_depth=training_config(:model_defaults, :decision_tree_reg).max_depth) =
        new(max_depth, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct RandomForestReg <: RegressionModel
    n_trees::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    RandomForestReg(; n_trees=training_config(:model_defaults, :random_forest_reg).n_trees) =
        new(n_trees, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct XGBoostReg <: RegressionModel
    num_round::Int
    max_depth::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    XGBoostReg(; num_round=training_config(:model_defaults, :xgboost_reg).num_round,
               max_depth=training_config(:model_defaults, :xgboost_reg).max_depth) =
        new(num_round, max_depth, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct KNNReg <: RegressionModel
    K::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    KNNReg(; K=training_config(:model_defaults, :knn_reg).K) =
        new(K, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct SVMReg <: RegressionModel
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    SVMReg() = new(false, nothing, Dict{Symbol, Vector{String}}())
end

# ============================================================================
# CLASSIFICATION MODELS (8)
# ============================================================================

mutable struct LogisticCls <: ClassificationModel
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    LogisticCls() = new(false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct DecisionTreeCls <: ClassificationModel
    max_depth::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    DecisionTreeCls(; max_depth=training_config(:model_defaults, :decision_tree_cls).max_depth) =
        new(max_depth, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct RandomForestCls <: ClassificationModel
    n_trees::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    RandomForestCls(; n_trees=training_config(:model_defaults, :random_forest_cls).n_trees) =
        new(n_trees, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct AdaBoostCls <: ClassificationModel
    n_iter::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    AdaBoostCls(; n_iter=training_config(:model_defaults, :adaboost_cls).n_iter) =
        new(n_iter, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct XGBoostCls <: ClassificationModel
    num_round::Int
    max_depth::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    XGBoostCls(; num_round=training_config(:model_defaults, :xgboost_cls).num_round,
               max_depth=training_config(:model_defaults, :xgboost_cls).max_depth) =
        new(num_round, max_depth, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct KNNCls <: ClassificationModel
    K::Int
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    KNNCls(; K=training_config(:model_defaults, :knn_cls).K) =
        new(K, false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct SVMCls <: ClassificationModel
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    SVMCls() = new(false, nothing, Dict{Symbol, Vector{String}}())
end

mutable struct NaiveBayesCls <: ClassificationModel
    is_trained::Bool
    machine::Any
    _feature_levels::Dict{Symbol, Vector{String}}
    NaiveBayesCls() = new(false, nothing, Dict{Symbol, Vector{String}}())
end

# ============================================================================
# TIME SERIES — STATISTICAL MODELS (2)
# ============================================================================

mutable struct ARIMAModel <: StatTimeSeriesModel
    order::Tuple{Int,Int,Int}
    is_trained::Bool
    _model::Any
    _train_data::Vector{Float64}
    ARIMAModel(; order=training_config(:model_defaults, :arima).order) =
        new(order, false, nothing, Float64[])
end

mutable struct ETSModel <: StatTimeSeriesModel
    is_trained::Bool
    _model::Any
    _train_data::Vector{Float64}
    ETSModel() = new(false, nothing, Float64[])
end

# ============================================================================
# TIME SERIES — DEEP LEARNING MODELS (3)
# ============================================================================

mutable struct RNNModel <: DeepTimeSeriesModel
    input_dim::Int
    hidden_dim::Int
    epochs::Int
    seq_len::Int
    is_trained::Bool
    _chain::Union{SeqChain, Nothing}
    _train_data::Vector{Float64}
    _y_mean::Float64
    _y_std::Float64
    RNNModel(input_dim; hidden_dim=training_config(:model_defaults, :deep_ts).hidden_dim,
             epochs=training_config(:model_defaults, :deep_ts).epochs,
             seq_len=training_config(:model_defaults, :deep_ts).seq_len) =
        new(input_dim, hidden_dim, epochs, seq_len, false, nothing, Float64[], 0.0, 1.0)
end

mutable struct LSTMModel <: DeepTimeSeriesModel
    input_dim::Int
    hidden_dim::Int
    epochs::Int
    seq_len::Int
    is_trained::Bool
    _chain::Union{SeqChain, Nothing}
    _train_data::Vector{Float64}
    _y_mean::Float64
    _y_std::Float64
    LSTMModel(input_dim; hidden_dim=training_config(:model_defaults, :deep_ts).hidden_dim,
              epochs=training_config(:model_defaults, :deep_ts).epochs,
              seq_len=training_config(:model_defaults, :deep_ts).seq_len) =
        new(input_dim, hidden_dim, epochs, seq_len, false, nothing, Float64[], 0.0, 1.0)
end

mutable struct GRUModel <: DeepTimeSeriesModel
    input_dim::Int
    hidden_dim::Int
    epochs::Int
    seq_len::Int
    is_trained::Bool
    _chain::Union{SeqChain, Nothing}
    _train_data::Vector{Float64}
    _y_mean::Float64
    _y_std::Float64
    GRUModel(input_dim; hidden_dim=training_config(:model_defaults, :deep_ts).hidden_dim,
             epochs=training_config(:model_defaults, :deep_ts).epochs,
             seq_len=training_config(:model_defaults, :deep_ts).seq_len) =
        new(input_dim, hidden_dim, epochs, seq_len, false, nothing, Float64[], 0.0, 1.0)
end
