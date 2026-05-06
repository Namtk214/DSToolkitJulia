# ==============================================================================
# DSToolkit — predict Methods (Proper Multiple Dispatch)
# ==============================================================================

# --- Guard: ensure model is trained ---
function _assert_trained(model::AbstractToolkitModel)
    if !model.is_trained
        name = typeof(model).name.name
        error("$name is not trained. Call fit!(model, ...) first.")
    end
end

# ==============================================================================
# Tabular Models — Regression
# ==============================================================================

"""
    predict(model::RegressionModel, X::DataFrame) → Vector{Float64}

Predict continuous values for tabular regression models.
"""
function predict(model::RegressionModel, X::DataFrame)
    _assert_trained(model)
    if typeof(model.machine.model) <: MLJ.Probabilistic
        preds = MLJ.predict_mean(model.machine, X)
    else
        preds = MLJ.predict(model.machine, X)
    end
    return Float64.(preds)
end

# ==============================================================================
# Tabular Models — Classification
# ==============================================================================

"""
    predict(model::ClassificationModel, X::DataFrame) → CategoricalVector

Predict class labels for tabular classification models.
Returns the mode (most likely class) of the predicted distribution.
"""
function predict(model::ClassificationModel, X::DataFrame)
    _assert_trained(model)
    if typeof(model.machine.model) <: MLJ.Probabilistic
        preds = MLJ.predict_mode(model.machine, X)
    else
        preds = MLJ.predict(model.machine, X)
    end
    return preds
end

"""
    predict_proba(model::ClassificationModel, X::DataFrame) → Matrix

Return class probability distributions for classification models.
"""
function predict_proba(model::ClassificationModel, X::DataFrame)
    _assert_trained(model)
    return MLJ.predict(model.machine, X)
end

# ==============================================================================
# Statistical Time Series
# ==============================================================================

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

# ==============================================================================
# Deep Learning Time Series
# ==============================================================================

"""
    predict(model::DeepTimeSeriesModel, X::AbstractArray{T,3}) → Matrix{Float32}

Run inference on a deep TS model. Input `X` shape: `(features, seq_len, batch)`.
Returns `(output_dim, batch)`.
"""
function predict(model::DeepTimeSeriesModel, X::AbstractArray{T,3}) where T
    _assert_trained(model)
    return model._chain(Float32.(X))
end
