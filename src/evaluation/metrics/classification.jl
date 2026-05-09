# Classification Evaluation Metrics

"""
    evaluate(model::ClassificationModel, X_test::DataFrame, y_test) → Dict

Evaluate a classification model returning Accuracy, Macro-Precision,
Macro-Recall, and Macro-F1.
"""
function evaluate(model::ClassificationModel, X_test::DataFrame, y_test::AbstractVector)
    preds = predict(model, X_test)
    return classification_metrics(preds, y_test)
end

"""
    classification_metrics(preds, truth) → Dict

Calculate classification metrics: Accuracy, Macro-Precision, Macro-Recall, Macro-F1.
"""
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
