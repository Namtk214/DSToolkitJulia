# Time Series Evaluation Metrics

"""
    evaluate(model::StatTimeSeriesModel, y_test::AbstractVector) → Dict

Evaluate a statistical TS model by forecasting `length(y_test)` steps ahead.
"""
function evaluate(model::StatTimeSeriesModel, y_test::AbstractVector)
    preds = predict(model, length(y_test))
    return timeseries_metrics(preds, Float64.(y_test))
end

"""
    evaluate(model::DeepTimeSeriesModel, X_test, y_test) → Dict

Evaluate a deep TS model on test sequences.
"""
function evaluate(model::DeepTimeSeriesModel, X_test::AbstractArray{T,3},
                  y_test::AbstractArray) where T
    preds = predict(model, X_test)
    return timeseries_metrics(vec(collect(Float64.(preds))), vec(collect(Float64.(y_test))))
end

"""
    evaluate(model::DeepTimeSeriesModel, y_test::AbstractVector) → Dict

Evaluate an autoregressive deep TS model trained from a univariate series.
"""
function evaluate(model::DeepTimeSeriesModel, y_test::AbstractVector)
    preds = predict(model, length(y_test))
    return timeseries_metrics(preds, Float64.(y_test))
end

"""
    timeseries_metrics(preds, truth) → Dict

Calculate time series metrics: RMSE, MAE, MAPE.
"""
function timeseries_metrics(preds::AbstractVector, truth::AbstractVector)
    residuals = Float64.(preds) .- Float64.(truth)
    n = length(residuals)

    rmse = sqrt(sum(residuals .^ 2) / n)
    mae = sum(abs.(residuals)) / n

    nonzero_mask = Float64.(truth) .!= 0.0
    mape = if any(nonzero_mask)
        mean(abs.(residuals[nonzero_mask] ./ Float64.(truth)[nonzero_mask])) * 100.0
    else
        NaN
    end

    return Dict{String,Float64}("RMSE" => rmse, "MAE" => mae, "MAPE" => mape)
end
