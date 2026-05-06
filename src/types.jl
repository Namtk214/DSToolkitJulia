# --- Abstract Type Hierarchy ---
abstract type AbstractToolkitModel end
abstract type TabularModel <: AbstractToolkitModel end
abstract type RegressionModel <: TabularModel end
abstract type ClassificationModel <: TabularModel end
abstract type TimeSeriesModel <: AbstractToolkitModel end
abstract type DeepTimeSeriesModel <: TimeSeriesModel end
abstract type StatTimeSeriesModel <: TimeSeriesModel end

# --- Standardized Data Container ---
"""
    ToolkitData

Standardized container returned by `ingest_data()`.
- `X`: Feature DataFrame (or `nothing` for univariate time series)
- `y`: Target vector
- `task`: Detected task type — `:regression`, `:classification`, or `:timeseries`
"""
struct ToolkitData
    X::Union{DataFrame, Nothing}
    y::AbstractVector
    task::Symbol
end

# --- Comparison Result Container ---
"""
    ComparisonResult

Returned by `auto_compare()`. Contains the full results table, the best model,
and all trained models for the user to pick from.
"""
struct ComparisonResult
    task::Symbol
    results::DataFrame
    best_model::AbstractToolkitModel
    best_model_name::String
    all_models::Vector{Pair{String, AbstractToolkitModel}}
end

# Regression Models (9)
mutable struct LinearReg <: RegressionModel
    is_trained::Bool
    machine::Any
    LinearReg() = new(false, nothing)
end

mutable struct RidgeReg <: RegressionModel
    lambda::Float64
    is_trained::Bool
    machine::Any
    RidgeReg(; lambda=1.0) = new(lambda, false, nothing)
end

mutable struct LassoReg <: RegressionModel
    lambda::Float64
    is_trained::Bool
    machine::Any
    LassoReg(; lambda=1.0) = new(lambda, false, nothing)
end

mutable struct ElasticNetReg <: RegressionModel
    lambda::Float64
    alpha::Float64
    is_trained::Bool
    machine::Any
    ElasticNetReg(; lambda=1.0, alpha=0.5) = new(lambda, alpha, false, nothing)
end

mutable struct DecisionTreeReg <: RegressionModel
    max_depth::Int
    is_trained::Bool
    machine::Any
    DecisionTreeReg(; max_depth=-1) = new(max_depth, false, nothing)
end

mutable struct RandomForestReg <: RegressionModel
    n_trees::Int
    is_trained::Bool
    machine::Any
    RandomForestReg(; n_trees=100) = new(n_trees, false, nothing)
end

mutable struct XGBoostReg <: RegressionModel
    num_round::Int
    max_depth::Int
    is_trained::Bool
    machine::Any
    XGBoostReg(; num_round=100, max_depth=6) = new(num_round, max_depth, false, nothing)
end

mutable struct KNNReg <: RegressionModel
    K::Int
    is_trained::Bool
    machine::Any
    KNNReg(; K=5) = new(K, false, nothing)
end

mutable struct SVMReg <: RegressionModel
    is_trained::Bool
    machine::Any
    SVMReg() = new(false, nothing)
end

# Classification Models (8)
mutable struct LogisticCls <: ClassificationModel
    is_trained::Bool
    machine::Any
    LogisticCls() = new(false, nothing)
end

mutable struct DecisionTreeCls <: ClassificationModel
    max_depth::Int
    is_trained::Bool
    machine::Any
    DecisionTreeCls(; max_depth=-1) = new(max_depth, false, nothing)
end

mutable struct RandomForestCls <: ClassificationModel
    n_trees::Int
    is_trained::Bool
    machine::Any
    RandomForestCls(; n_trees=100) = new(n_trees, false, nothing)
end

mutable struct AdaBoostCls <: ClassificationModel
    n_iter::Int
    is_trained::Bool
    machine::Any
    AdaBoostCls(; n_iter=10) = new(n_iter, false, nothing)
end

mutable struct XGBoostCls <: ClassificationModel
    num_round::Int
    max_depth::Int
    is_trained::Bool
    machine::Any
    XGBoostCls(; num_round=100, max_depth=6) = new(num_round, max_depth, false, nothing)
end

mutable struct KNNCls <: ClassificationModel
    K::Int
    is_trained::Bool
    machine::Any
    KNNCls(; K=5) = new(K, false, nothing)
end

mutable struct SVMCls <: ClassificationModel
    is_trained::Bool
    machine::Any
    SVMCls() = new(false, nothing)
end

mutable struct NaiveBayesCls <: ClassificationModel
    is_trained::Bool
    machine::Any
    NaiveBayesCls() = new(false, nothing)
end

# Time Series — Statistical Models (2)
mutable struct ARIMAModel <: StatTimeSeriesModel
    order::Tuple{Int,Int,Int}
    is_trained::Bool
    _model::Any
    _train_data::Vector{Float64}
    ARIMAModel(; order=(1,1,1)) = new(order, false, nothing, Float64[])
end

mutable struct ETSModel <: StatTimeSeriesModel
    is_trained::Bool
    _model::Any
    _train_data::Vector{Float64}
    ETSModel() = new(false, nothing, Float64[])
end


# Time Series — Deep Learning Models (3)

"""
    SeqChain

Custom Flux model that properly handles sequence input for RNN/LSTM/GRU.
Input shape: `(features, seq_len, batch)`.
Iterates timesteps through the recurrent layer, takes the final hidden state,
and passes it through a dense output layer.
"""
struct SeqChain
    rnn_layer
    dense_layer
end

Flux.@layer SeqChain

function (m::SeqChain)(X::AbstractArray{T,3}) where T
    seq_len = size(X, 2)
    h = nothing
    for t in 1:seq_len
        h = m.rnn_layer(X[:, t, :])
    end
    return m.dense_layer(h)
end

mutable struct RNNModel <: DeepTimeSeriesModel
    input_dim::Int
    hidden_dim::Int
    epochs::Int
    is_trained::Bool
    _chain::Union{SeqChain, Nothing}
    RNNModel(input_dim; hidden_dim=32, epochs=50) = new(input_dim, hidden_dim, epochs, false, nothing)
end

mutable struct LSTMModel <: DeepTimeSeriesModel
    input_dim::Int
    hidden_dim::Int
    epochs::Int
    is_trained::Bool
    _chain::Union{SeqChain, Nothing}
    LSTMModel(input_dim; hidden_dim=32, epochs=50) = new(input_dim, hidden_dim, epochs, false, nothing)
end

mutable struct GRUModel <: DeepTimeSeriesModel
    input_dim::Int
    hidden_dim::Int
    epochs::Int
    is_trained::Bool
    _chain::Union{SeqChain, Nothing}
    GRUModel(input_dim; hidden_dim=32, epochs=50) = new(input_dim, hidden_dim, epochs, false, nothing)
end
