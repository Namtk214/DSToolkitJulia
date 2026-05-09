
"""
    auto_compare(data::ToolkitData; ratio=0.8) → ComparisonResult

Automatically split data, train ALL suitable models for the detected task,
evaluate each one, and display a comparison table. Returns a `ComparisonResult`
containing the best model and all results.
"""
function auto_compare(data::ToolkitData; ratio=0.8)
    if data.task == :timeseries
        y_train, y_test = train_test_split(data.y; ratio=ratio)
        return _compare_stat_ts(y_train, y_test)
    else
        X_train, X_test, y_train, y_test = train_test_split(data.X, data.y; ratio=ratio)
        if data.task == :regression
            return _compare_regression(X_train, y_train, X_test, y_test)
        else
            return _compare_classification(X_train, y_train, X_test, y_test)
        end
    end
end

"""
    auto_compare(X_train::DataFrame, y_train, X_test, y_test; task=:auto)

Explicit variant — train all models for the given task and compare.
If `task=:auto`, it will be auto-detected from `y_train`.
"""
function auto_compare(X_train::DataFrame, y_train::AbstractVector,
                      X_test::DataFrame, y_test::AbstractVector;
                      task::Symbol=:auto)
    detected = task == :auto ? detect_task(y_train) : task
    if detected == :regression
        return _compare_regression(X_train, y_train, X_test, y_test)
    else
        return _compare_classification(X_train, y_train, X_test, y_test)
    end
end

"""
    auto_compare(y_train::AbstractVector, y_test::AbstractVector)

Compare statistical time series models (ARIMA, ETS).
"""
function auto_compare(y_train::AbstractVector, y_test::AbstractVector)
    return _compare_stat_ts(y_train, y_test)
end

"""
    auto_compare(X_train::AbstractArray{T,3}, y_train, X_test, y_test)

Compare deep learning time series models (RNN, LSTM, GRU).
"""
function auto_compare(X_train::AbstractArray{T,3}, y_train::AbstractArray,
                      X_test::AbstractArray{T,3}, y_test::AbstractArray) where T
    return _compare_deep_ts(X_train, y_train, X_test, y_test)
end

# Internal Comparison Runners


function _compare_regression(X_train, y_train, X_test, y_test)
    models = [
        "Linear Regression"  => LinearReg(),
        "Ridge Regression"   => RidgeReg(),
        "Lasso Regression"   => LassoReg(),
        "Elastic Net"        => ElasticNetReg(),
        "Decision Tree"      => DecisionTreeReg(),
        "Random Forest"      => RandomForestReg(),
        "XGBoost"            => XGBoostReg(),
        "KNN"                => KNNReg(),
        "SVM"                => SVMReg(),
    ]
    return _run_tabular_comparison(models, X_train, y_train, X_test, y_test, :regression)
end

function _compare_classification(X_train, y_train, X_test, y_test)
    models = [
        "Logistic Regression" => LogisticCls(),
        "Decision Tree"       => DecisionTreeCls(),
        "Random Forest"       => RandomForestCls(),
        "AdaBoost"            => AdaBoostCls(),
        "XGBoost"             => XGBoostCls(),
        "KNN"                 => KNNCls(),
        "SVM"                 => SVMCls(),
        "Naive Bayes"         => NaiveBayesCls(),
    ]
    return _run_tabular_comparison(models, X_train, y_train, X_test, y_test, :classification)
end

function _run_tabular_comparison(models, X_train, y_train, X_test, y_test, task)
    if task == :regression
        metric_keys = ["RMSE", "MAE", "R²", "MAPE"]
        sort_col = "RMSE"
        sort_rev = false  # lower RMSE is better
    else
        metric_keys = ["Accuracy", "Macro-F1", "Macro-Precision", "Macro-Recall"]
        sort_col = "Accuracy"
        sort_rev = true   # higher accuracy is better
    end

    trained_models = Pair{String, AbstractToolkitModel}[]
    rows = []

    println("\n" * "="^60)
    println("  🔄 Training $(length(models)) $(task) models...")
    println("="^60)

    for (name, model) in models
        try
            fit!(model, X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            row = Dict{String,Any}("Model" => name)
            for k in metric_keys
                row[k] = get(metrics, k, NaN)
            end
            push!(rows, row)
            push!(trained_models, name => model)
        catch e
            @warn "⚠ Failed to train $name" exception=(e, catch_backtrace())
            row = Dict{String,Any}("Model" => name)
            for k in metric_keys
                row[k] = NaN
            end
            push!(rows, row)
        end
    end

    result_df = _build_result_df(rows, metric_keys)

    # Sort by primary metric
    valid_mask = .!isnan.(result_df[!, sort_col])
    if any(valid_mask)
        sort!(result_df, sort_col, rev=sort_rev)
    end

    # Identify best
    if isempty(trained_models)
        error("All models failed to train. Check warnings above for details.")
    end
    best_name = result_df[1, "Model"]
    best_idx = findfirst(p -> p.first == best_name, trained_models)
    best_model = best_idx !== nothing ? trained_models[best_idx].second : trained_models[1].second

    # Display results
    _display_results(result_df, task, best_name)

    return ComparisonResult(task, result_df, best_model, best_name, trained_models)
end

function _compare_stat_ts(y_train, y_test)
    models = [
        "ARIMA" => ARIMAModel(),
        "ETS"   => ETSModel(),
    ]

    metric_keys = ["RMSE", "MAE", "MAPE"]
    trained_models = Pair{String, AbstractToolkitModel}[]
    rows = []

    println("\n" * "="^60)
    println("  🔄 Training $(length(models)) statistical time series models...")
    println("="^60)

    for (name, model) in models
        try
            fit!(model, y_train)
            metrics = evaluate(model, y_test)
            row = Dict{String,Any}("Model" => name)
            for k in metric_keys
                row[k] = get(metrics, k, NaN)
            end
            push!(rows, row)
            push!(trained_models, name => model)
        catch e
            @warn "⚠ Failed to train $name" exception=(e, catch_backtrace())
            row = Dict{String,Any}("Model" => name)
            for k in metric_keys; row[k] = NaN; end
            push!(rows, row)
        end
    end

    result_df = _build_result_df(rows, metric_keys)
    sort!(result_df, "RMSE")

    if isempty(trained_models)
        error("All models failed to train. Check warnings above for details.")
    end
    best_name = result_df[1, "Model"]
    best_idx = findfirst(p -> p.first == best_name, trained_models)
    best_model = best_idx !== nothing ? trained_models[best_idx].second : trained_models[1].second

    _display_results(result_df, :timeseries, best_name)

    return ComparisonResult(:timeseries, result_df, best_model, best_name, trained_models)
end

function _compare_deep_ts(X_train, y_train, X_test, y_test)
    in_dim = size(X_train, 1)
    models = [
        "RNN"  => RNNModel(in_dim),
        "LSTM" => LSTMModel(in_dim),
        "GRU"  => GRUModel(in_dim),
    ]

    metric_keys = ["RMSE", "MAE", "MAPE"]
    trained_models = Pair{String, AbstractToolkitModel}[]
    rows = []

    println("\n" * "="^60)
    println("  🔄 Training $(length(models)) deep learning time series models...")
    println("="^60)

    for (name, model) in models
        try
            fit!(model, X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            row = Dict{String,Any}("Model" => name)
            for k in metric_keys
                row[k] = get(metrics, k, NaN)
            end
            push!(rows, row)
            push!(trained_models, name => model)
        catch e
            @warn "⚠ Failed to train $name" exception=(e, catch_backtrace())
            row = Dict{String,Any}("Model" => name)
            for k in metric_keys; row[k] = NaN; end
            push!(rows, row)
        end
    end

    result_df = _build_result_df(rows, metric_keys)
    sort!(result_df, "RMSE")

    if isempty(trained_models)
        error("All models failed to train. Check warnings above for details.")
    end
    best_name = result_df[1, "Model"]
    best_idx = findfirst(p -> p.first == best_name, trained_models)
    best_model = best_idx !== nothing ? trained_models[best_idx].second : trained_models[1].second

    _display_results(result_df, :timeseries_deep, best_name)

    return ComparisonResult(:timeseries_deep, result_df, best_model, best_name, trained_models)
end

# Result DataFrame Builder

function _build_result_df(rows::Vector, metric_keys::Vector{String})
    model_names = String[r["Model"] for r in rows]
    df = DataFrame("Model" => model_names)
    for k in metric_keys
        df[!, k] = Float64[get(r, k, NaN) for r in rows]
    end
    return df
end
