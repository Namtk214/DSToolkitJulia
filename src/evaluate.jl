# Regression Metrics

"""
    evaluate(model::RegressionModel, X_test::DataFrame, y_test) → Dict

Evaluate a regression model returning RMSE, MAE, R², and MAPE.
"""
function evaluate(model::RegressionModel, X_test::DataFrame, y_test::AbstractVector)
    preds = predict(model, X_test)
    return regression_metrics(preds, y_test)
end

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

# Classification Metrics

"""
    evaluate(model::ClassificationModel, X_test::DataFrame, y_test) → Dict

Evaluate a classification model returning Accuracy, Macro-Precision,
Macro-Recall, and Macro-F1.
"""
function evaluate(model::ClassificationModel, X_test::DataFrame, y_test::AbstractVector)
    preds = predict(model, X_test)
    return classification_metrics(preds, y_test)
end

function classification_metrics(preds, truth)
    # Unwrap CategoricalValues to plain values for comparison
    preds_plain = _unwrap.(preds)
    truth_plain = _unwrap.(truth)

    accuracy = mean(preds_plain .== truth_plain)

    classes = sort(unique(vcat(preds_plain, truth_plain)))
    precisions = Float64[]
    recalls = Float64[]
    f1_scores = Float64[]

    for c in classes
        tp = sum((preds_plain .== c) .& (truth_plain .== c))
        fp = sum((preds_plain .== c) .& (truth_plain .!= c))
        fn = sum((preds_plain .!= c) .& (truth_plain .== c))

        prec = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
        rec  = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
        f1   = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0
        
        push!(precisions, prec)
        push!(recalls, rec)
        push!(f1_scores, f1)
    end

    macro_prec = mean(precisions)
    macro_rec  = mean(recalls)
    macro_f1   = mean(f1_scores)

    return Dict{String,Float64}(
        "Accuracy"  => accuracy,
        "Macro-Precision" => macro_prec,
        "Macro-Recall"    => macro_rec,
        "Macro-F1"        => macro_f1
    )
end

# Helper to unwrap CategoricalValue to its underlying value
_unwrap(x::CategoricalValue) = MLJ.unwrap(x)
_unwrap(x) = x

# Time Series Metrics

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
