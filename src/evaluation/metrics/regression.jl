# Regression Evaluation Metrics

"""
    evaluate(model::RegressionModel, X_test::DataFrame, y_test) → Dict

Evaluate a regression model returning RMSE, MAE, R², and MAPE.
"""
function evaluate(model::RegressionModel, X_test::DataFrame, y_test::AbstractVector)
    preds = predict(model, X_test)
    return regression_metrics(preds, y_test)
end

"""
    regression_metrics(preds, truth) → Dict

Calculate regression metrics: RMSE, MAE, R², MAPE.
"""
function regression_metrics(preds::AbstractVector, truth::AbstractVector)
    residuals = Float64.(preds) .- Float64.(truth)
    n = length(residuals)

    rmse = sqrt(sum(residuals .^ 2) / n)
    mae = sum(abs.(residuals)) / n

    ss_res = sum(residuals .^ 2)
    ss_tot = sum((Float64.(truth) .- mean(Float64.(truth))) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN

    nonzero_mask = Float64.(truth) .!= 0.0
    mape = if any(nonzero_mask)
        mean(abs.(residuals[nonzero_mask] ./ Float64.(truth)[nonzero_mask])) * 100.0
    else
        NaN
    end

    return Dict{String,Float64}(
        "RMSE" => rmse, "MAE" => mae, "R²" => r2, "MAPE" => mape
    )
end
