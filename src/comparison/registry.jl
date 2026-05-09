# Model Registry for Auto-Comparison

# Regression Models (9)
const REGRESSION_MODELS = [
    (:LinearReg, LinearReg()),
    (:RidgeReg, RidgeReg()),
    (:LassoReg, LassoReg()),
    (:ElasticNetReg, ElasticNetReg()),
    (:DecisionTreeReg, DecisionTreeReg()),
    (:RandomForestReg, RandomForestReg()),
    (:XGBoostReg, XGBoostReg()),
    (:KNNReg, KNNReg()),
    (:SVMReg, SVMReg())
]

# Classification Models (8)
const CLASSIFICATION_MODELS = [
    (:LogisticCls, LogisticCls()),
    (:DecisionTreeCls, DecisionTreeCls()),
    (:RandomForestCls, RandomForestCls()),
    (:AdaBoostCls, AdaBoostCls()),
    (:XGBoostCls, XGBoostCls()),
    (:KNNCls, KNNCls()),
    (:SVMCls, SVMCls()),
    (:NaiveBayesCls, NaiveBayesCls())
]

# Statistical Time Series Models (2)
const STAT_TS_MODELS = [
    (:ARIMA, ARIMAModel()),
    (:ETS, ETSModel())
]

# Deep Learning Time Series Models (3)
function get_deep_ts_models(input_dim::Int)
    return [
        (:RNN, RNNModel(input_dim)),
        (:LSTM, LSTMModel(input_dim)),
        (:GRU, GRUModel(input_dim))
    ]
end

"""
    get_models_for_task(task::Symbol, input_dim::Union{Int,Nothing}=nothing)

Get list of models for a specific task.
Returns Vector of (name::Symbol, model::AbstractToolkitModel) pairs.
"""
function get_models_for_task(task::Symbol, input_dim::Union{Int,Nothing}=nothing)
    if task == :regression
        return REGRESSION_MODELS
    elseif task == :classification
        return CLASSIFICATION_MODELS
    elseif task == :timeseries
        return STAT_TS_MODELS
    elseif task == :deep_timeseries
        input_dim === nothing && error("input_dim required for deep time series models")
        return get_deep_ts_models(input_dim)
    else
        error("Unknown task: $task")
    end
end
