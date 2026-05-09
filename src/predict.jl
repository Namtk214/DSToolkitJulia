# --- Guard: ensure model is trained ---
function _assert_trained(model::AbstractToolkitModel)
    if !model.is_trained
        name = typeof(model).name.name
        error("$name is not trained. Call fit!(model, ...) first.")
    end
end

# Tabular Models — Regression

"""
    predict(model::RegressionModel, X::DataFrame) → Vector{Float64}

Predict continuous values for tabular regression models.
"""
function predict(model::RegressionModel, X::DataFrame)
    _assert_trained(model)
    X_pred = _prepare_tabular_features(X, model._feature_levels)
    if typeof(model.machine.model) <: MLJ.Probabilistic
        preds = MLJ.predict_mean(model.machine, X_pred)
    else
        preds = MLJ.predict(model.machine, X_pred)
    end
    return Float64.(preds)
end

# Tabular Models — Classification

"""
    predict(model::ClassificationModel, X::DataFrame) → CategoricalVector

Predict class labels for tabular classification models.
Returns the mode (most likely class) of the predicted distribution.
"""
function predict(model::ClassificationModel, X::DataFrame)
    _assert_trained(model)
    X_pred = _prepare_tabular_features(X, model._feature_levels)
    if typeof(model.machine.model) <: MLJ.Probabilistic
        preds = MLJ.predict_mode(model.machine, X_pred)
    else
        preds = MLJ.predict(model.machine, X_pred)
    end
    return preds
end

"""
    predict_proba(model::ClassificationModel, X::DataFrame) → Matrix

Return class probability distributions for classification models.
"""
function predict_proba(model::ClassificationModel, X::DataFrame)
    _assert_trained(model)
    X_pred = _prepare_tabular_features(X, model._feature_levels)
    return MLJ.predict(model.machine, X_pred)
end

# Statistical Time Series

"""
    predict(model::StatTimeSeriesModel, steps::Int) → Vector{Float64}

Forecast `steps` time steps ahead using a fitted statistical TS model.
"""
function predict(model::StatTimeSeriesModel, steps::Int)
    _assert_trained(model)
    steps > 0 || error("Forecast steps must be positive, got $steps.")
    forecasted = SSM.forecast(model._model, steps)
    # forecasted.expected_value is usually a Vector{Vector{Float64}} for SSM
    return Float64[v[1] for v in forecasted.expected_value]
end

# Deep Learning Time Series

"""
    predict(model::DeepTimeSeriesModel, X::AbstractArray{T,3}) → Matrix{Float32}

Run inference on a deep TS model. Input `X` shape: `(features, seq_len, batch)`.
Returns `(output_dim, batch)`.
"""
function predict(model::DeepTimeSeriesModel, X::AbstractArray{T,3}) where T
    _assert_trained(model)
    return model._chain(Float32.(X))
end

"""
    predict(model::DeepTimeSeriesModel, steps::Int) → Vector{Float64}

Autoregressive forecast for deep time-series models trained with
`fit!(model, y; seq_len=...)`.
"""
function predict(model::DeepTimeSeriesModel, steps::Int)
    _assert_trained(model)
    steps > 0 || error("Forecast steps must be positive, got $steps.")
    length(model._train_data) >= model.seq_len || error("No training series stored for autoregressive forecasting. Train with fit!(model, y; seq_len=...) first.")

    history = collect(Float64, model._train_data)
    forecasts = Float64[]

    for _ in 1:steps
        window = history[end-model.seq_len+1:end]
        scaled = (window .- model._y_mean) ./ model._y_std
        X = reshape(Float32.(scaled), 1, model.seq_len, 1)
        pred_scaled = Float64(model._chain(X)[1, 1])
        pred = pred_scaled * model._y_std + model._y_mean
        push!(forecasts, pred)
        push!(history, pred)
    end

    return forecasts
end
