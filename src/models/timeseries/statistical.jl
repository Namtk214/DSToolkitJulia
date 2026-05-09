# Statistical Time Series Models
# Includes: ARIMAModel, ETSModel

function fit!(model::ARIMAModel, y::AbstractVector)
    validate_timeseries(y)
    y_float = Float64.(y)
    ssm_model = SSM.SARIMA(y_float; order=model.order)
    SSM.fit!(ssm_model)
    model._model = ssm_model
    model._train_data = y_float
    model.is_trained = true
    @info "✓ ARIMA$(model.order) trained on $(length(y)) points"
    return model
end

function fit!(model::ETSModel, y::AbstractVector)
    validate_timeseries(y)
    y_float = Float64.(y)
    ssm_model = SSM.ExponentialSmoothing(y_float)
    SSM.fit!(ssm_model)
    model._model = ssm_model
    model._train_data = y_float
    model.is_trained = true
    @info "✓ ETS trained on $(length(y)) points"
    return model
end
