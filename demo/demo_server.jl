# DSToolkit Demo Server
# Simple HTTP server for demonstration without Genie dependency

using HTTP
using JSON
using DataFrames
using Statistics
using DSToolkit
import MLJ

const DATASETS = Dict(
    "iris" => (label="Iris", task=:classification, loader=load_iris),
    "titanic" => (label="Titanic", task=:classification, loader=load_titanic),
    "wine_quality" => (label="Wine Quality", task=:classification, loader=load_wine_quality),
    "housing" => (label="Housing", task=:regression, loader=load_housing),
    "diabetes" => (label="Diabetes", task=:regression, loader=load_diabetes),
    "synthetic_reg" => (label="Synthetic Regression", task=:regression, loader=load_synthetic_reg),
    "airline_passengers" => (label="Airline Passengers", task=:timeseries, loader=load_airline_passengers),
    "stock_prices" => (label="Stock Prices", task=:timeseries, loader=load_stock_prices),
    "temperature" => (label="Temperature", task=:timeseries, loader=load_temperature),
)

const TARGET_LABELS = Dict(
    "titanic" => Dict(
        "0" => "Not survived",
        "1" => "Survived",
    ),
    "wine_quality" => Dict(
        "3" => "Quality 3",
        "4" => "Quality 4",
        "5" => "Quality 5",
        "6" => "Quality 6",
        "7" => "Quality 7",
        "8" => "Quality 8",
    ),
)

const DATASET_ORDER = [
    "iris", "titanic", "wine_quality",
    "housing", "diabetes", "synthetic_reg",
    "airline_passengers", "stock_prices", "temperature",
]

const DATASET_DESCRIPTIONS = Dict(
    "iris" => "150 iris flowers across 3 species (Setosa, Versicolor, Virginica). Features are sepal and petal dimensions in centimetres. A classic multiclass classification benchmark introduced by Ronald Fisher in 1936.",
    "titanic" => "891 passengers from the RMS Titanic disaster (April 1912). Features include ticket class, sex, age, number of siblings/spouses, and fare paid. Target: survival (0 = died, 1 = survived). Demonstrates how socioeconomic factors influenced survival odds.",
    "wine_quality" => "1,599 Portuguese red wine samples rated by expert tasters on quality from 3 to 8. Features are 11 physicochemical properties such as acidity, residual sugar, chlorides, alcohol content, and sulphates.",
    "housing" => "506 census tracts in the Boston area (1978) with 13 socioeconomic and geographic attributes — crime rate per capita, average rooms per dwelling, pupil-teacher ratio, proximity to employment centres, etc. Target: median home value in thousands of USD.",
    "diabetes" => "442 diabetes patients with 10 standardised baseline clinical measurements (age, BMI, blood pressure, and six serum values). Target is a continuous quantitative measure of disease progression one year after baseline.",
    "synthetic_reg" => "1,000 synthetically generated samples with a known linear + Gaussian-noise structure across multiple numeric features. Designed for clean, reproducible benchmarking of regression algorithms without real-world noise.",
    "airline_passengers" => "Monthly count of international airline passengers from January 1949 to December 1960 — 144 time steps in total. Exhibits strong multiplicative seasonality and an upward trend, making it a canonical time series forecasting benchmark.",
    "stock_prices" => "252 daily synthetic stock price records simulating one full trading year. Shows realistic features such as trending, volatility clustering, and random noise — useful for evaluating financial forecasting models.",
    "temperature" => "365 daily temperature readings spanning a full calendar year. Captures clear seasonal cycles (summer peaks, winter troughs) and is well-suited for testing seasonal decomposition and forecasting methods.",
)

const DEMO_CONFIG = training_config(:demo)

const MODEL_GROUPS = Dict(
    :classification => [
        (id="logistic_cls", label="Logistic Regression", maker=() -> LogisticCls(), trainable=true),
        (id="decision_tree_cls", label="Decision Tree", maker=() -> DecisionTreeCls(), trainable=true),
        (id="random_forest_cls", label="Random Forest", maker=() -> RandomForestCls(n_trees=DEMO_CONFIG.random_forest.n_trees), trainable=true),
        (id="adaboost_cls", label="AdaBoost Stump", maker=() -> AdaBoostCls(n_iter=DEMO_CONFIG.adaboost.n_iter), trainable=true),
        (id="xgboost_cls", label="XGBoost", maker=() -> XGBoostCls(num_round=DEMO_CONFIG.xgboost.num_round, max_depth=DEMO_CONFIG.xgboost.max_depth), trainable=true),
        (id="knn_cls", label="KNN", maker=() -> KNNCls(K=DEMO_CONFIG.knn.K), trainable=true),
        (id="svm_cls", label="SVM", maker=() -> SVMCls(), trainable=true),
        (id="naive_bayes_cls", label="Gaussian Naive Bayes", maker=() -> NaiveBayesCls(), trainable=true),
    ],
    :regression => [
        (id="linear_reg", label="Linear Regression", maker=() -> LinearReg(), trainable=true),
        (id="ridge_reg", label="Ridge Regression", maker=() -> RidgeReg(lambda=DEMO_CONFIG.ridge.lambda), trainable=true),
        (id="lasso_reg", label="Lasso Regression", maker=() -> LassoReg(lambda=DEMO_CONFIG.lasso.lambda), trainable=true),
        (id="elastic_net_reg", label="Elastic Net", maker=() -> ElasticNetReg(lambda=DEMO_CONFIG.elastic_net.lambda, alpha=DEMO_CONFIG.elastic_net.alpha), trainable=true),
        (id="decision_tree_reg", label="Decision Tree", maker=() -> DecisionTreeReg(), trainable=true),
        (id="random_forest_reg", label="Random Forest", maker=() -> RandomForestReg(n_trees=DEMO_CONFIG.random_forest.n_trees), trainable=true),
        (id="xgboost_reg", label="XGBoost", maker=() -> XGBoostReg(num_round=DEMO_CONFIG.xgboost.num_round, max_depth=DEMO_CONFIG.xgboost.max_depth), trainable=true),
        (id="knn_reg", label="KNN", maker=() -> KNNReg(K=DEMO_CONFIG.knn.K), trainable=true),
        (id="svm_reg", label="SVM", maker=() -> SVMReg(), trainable=true),
    ],
    :timeseries => [
        (id="arima", label="ARIMA", maker=() -> ARIMAModel(order=DEMO_CONFIG.arima.order), trainable=true),
        (id="ets", label="ETS", maker=() -> ETSModel(), trainable=true),
        (id="rnn", label="RNN", maker=() -> RNNModel(DEMO_CONFIG.deep_ts.input_dim; hidden_dim=DEMO_CONFIG.deep_ts.hidden_dim, epochs=DEMO_CONFIG.deep_ts.epochs, seq_len=DEMO_CONFIG.deep_ts.seq_len), trainable=true),
        (id="lstm", label="LSTM", maker=() -> LSTMModel(DEMO_CONFIG.deep_ts.input_dim; hidden_dim=DEMO_CONFIG.deep_ts.hidden_dim, epochs=DEMO_CONFIG.deep_ts.epochs, seq_len=DEMO_CONFIG.deep_ts.seq_len), trainable=true),
        (id="gru", label="GRU", maker=() -> GRUModel(DEMO_CONFIG.deep_ts.input_dim; hidden_dim=DEMO_CONFIG.deep_ts.hidden_dim, epochs=DEMO_CONFIG.deep_ts.epochs, seq_len=DEMO_CONFIG.deep_ts.seq_len), trainable=true),
    ],
)

current_model = nothing
current_dataset_id = ""
current_dataset_name = ""
current_model_id = ""
current_model_name = ""
current_task = :classification
current_data = nothing
current_metrics = Dict{String, Any}()

function html_response(content)
    return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], body=content)
end

function json_response(data; status=200)
    return HTTP.Response(status, ["Content-Type" => "application/json"], body=JSON.json(data))
end

function find_model(task::Symbol, model_id::String)
    for spec in MODEL_GROUPS[task]
        spec.id == model_id && return spec
    end
    error("Model '$model_id' is not available for task '$task'.")
end

function default_model_id(task::Symbol)
    task == :classification && return "random_forest_cls"
    task == :regression && return "random_forest_reg"
    return "ets"
end

function display_label(dataset_id::String, value)
    raw = string(value)
    return get(get(TARGET_LABELS, dataset_id, Dict{String, String}()), raw, raw)
end

function display_counts(dataset_id::String, values)
    counts = Dict{String, Int}()
    for value in values
        label = display_label(dataset_id, value)
        counts[label] = get(counts, label, 0) + 1
    end
    return counts
end

function metric_summary(metrics, task::Symbol)
    if task == :classification
        return Dict("Accuracy" => get(metrics, "Accuracy", nothing), "Macro-F1" => get(metrics, "Macro-F1", nothing))
    elseif task == :regression || task == :timeseries
        return Dict("RMSE" => get(metrics, "RMSE", nothing), "MAE" => get(metrics, "MAE", nothing))
    end
    return Dict{String, Any}()
end

function train_demo_model(dataset_id::String="iris", model_id::String=default_model_id(DATASETS[dataset_id].task))
    global current_model, current_dataset_id, current_dataset_name, current_model_id
    global current_model_name, current_task, current_data, current_metrics

    haskey(DATASETS, dataset_id) || error("Unknown dataset '$dataset_id'.")
    dataset_spec = DATASETS[dataset_id]
    model_spec = find_model(dataset_spec.task, model_id)
    model_spec.trainable || error("$(model_spec.label) is listed in DSToolkit, but is not trainable in this demo.")

    println("Setting up demo model: $(dataset_spec.label) + $(model_spec.label)")

    data = dataset_spec.loader()
    model = model_spec.maker()
    metrics = Dict{String, Any}()

    if dataset_spec.task == :timeseries
        y_train, y_test = train_test_split(data.y; ratio=0.8)
        fit!(model, y_train)
        metrics = evaluate(model, y_test)
    else
        X_train, X_test, y_train, y_test = train_test_split(data; ratio=0.8)
        fit!(model, X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
    end

    current_model = model
    current_dataset_id = dataset_id
    current_dataset_name = dataset_spec.label
    current_model_id = model_id
    current_model_name = model_spec.label
    current_task = dataset_spec.task
    current_data = data
    current_metrics = Dict{String, Any}(string(k) => v for (k, v) in metrics)

    println("Demo model ready")
    println("   Dataset: $current_dataset_name")
    println("   Model: $current_model_name")
    println("   Metrics: $(metric_summary(current_metrics, current_task))")

    return current_metrics
end

function catalog_payload()
    datasets = [
        Dict(
            "id" => id,
            "label" => DATASETS[id].label,
            "task" => string(DATASETS[id].task),
        )
        for id in DATASET_ORDER
    ]

    models = Dict(
        string(task) => [
            Dict("id" => spec.id, "label" => spec.label, "trainable" => spec.trainable)
            for spec in specs
        ]
        for (task, specs) in MODEL_GROUPS
    )

    return Dict("datasets" => datasets, "models" => models)
end

function feature_schema(data=current_data, task=current_task)
    task == :timeseries && return []

    schema = []
    for col in names(data.X)
        vals = data.X[!, col]
        if eltype(vals) <: Number
            push!(schema, Dict(
                "name" => string(col),
                "kind" => "number",
                "min" => minimum(vals),
                "max" => maximum(vals),
                "mean" => mean(vals),
                "std" => std(vals),
            ))
        else
            levels = unique(string.(vals))
            push!(schema, Dict(
                "name" => string(col),
                "kind" => "category",
                "values" => levels,
                "default" => first(levels),
            ))
        end
    end
    return schema
end

function request_path_and_query(target)
    parts = split(target, "?"; limit=2)
    query = Dict{String, String}()
    if length(parts) == 2
        for pair in split(parts[2], "&")
            isempty(pair) && continue
            kv = split(pair, "="; limit=2)
            query[kv[1]] = length(kv) == 2 ? kv[2] : ""
        end
    end
    return parts[1], query
end

function counts_dict(values)
    labels = unique(string.(values))
    return Dict(label => count(==(label), string.(values)) for label in labels)
end

function histogram(values; bins=12)
    vals = Float64.(values)
    lo, hi = minimum(vals), maximum(vals)
    if lo == hi
        return Dict("edges" => [lo, hi], "counts" => [length(vals)])
    end
    width = (hi - lo) / bins
    counts = zeros(Int, bins)
    for value in vals
        idx = min(bins, max(1, floor(Int, (value - lo) / width) + 1))
        counts[idx] += 1
    end
    edges = [lo + width * i for i in 0:bins]
    return Dict("edges" => edges, "counts" => counts)
end

function numeric_summary(values)
    vals = Float64.(values)
    return Dict(
        "min" => minimum(vals),
        "q1" => quantile(vals, 0.25),
        "median" => median(vals),
        "mean" => mean(vals),
        "q3" => quantile(vals, 0.75),
        "max" => maximum(vals),
        "std" => std(vals),
    )
end

function correlation_matrix(df, numeric_cols)
    cols = string.(numeric_cols)
    matrix = [
        cor(Float64.(df[!, Symbol(a)]), Float64.(df[!, Symbol(b)]))
        for a in cols, b in cols
    ]
    return Dict("columns" => cols, "matrix" => matrix)
end

function visualization_payload(dataset_id)
    haskey(DATASETS, dataset_id) || error("Unknown dataset '$dataset_id'.")
    spec = DATASETS[dataset_id]
    data = spec.loader()

    if spec.task == :timeseries
        y = Float64.(data.y)
        n_sample = min(20, length(y))
        ts_samples = Dict(
            "columns" => ["#", "value"],
            "rows" => [Dict("#" => i, "value" => round(y[i]; digits=3)) for i in 1:n_sample],
        )
        return Dict(
            "dataset_id" => dataset_id,
            "dataset" => spec.label,
            "task" => string(spec.task),
            "description" => get(DATASET_DESCRIPTIONS, dataset_id, ""),
            "n_samples" => length(y),
            "n_features" => 0,
            "series" => y,
            "series_stats" => numeric_summary(y),
            "series_histogram" => histogram(y),
            "samples" => ts_samples,
        )
    end

    schema = feature_schema(data, spec.task)
    numeric_cols = [item["name"] for item in schema if item["kind"] == "number"]
    categorical_cols = [item["name"] for item in schema if item["kind"] == "category"]
    numeric = Dict(
        name => Dict(
            "summary" => numeric_summary(data.X[!, Symbol(name)]),
            "histogram" => histogram(data.X[!, Symbol(name)]),
        )
        for name in numeric_cols
    )
    categorical = Dict(
        name => Dict("counts" => counts_dict(data.X[!, Symbol(name)]))
        for name in categorical_cols
    )

    scatter = []
    if !isempty(numeric_cols)
        xcol = numeric_cols[1]
        ycol = length(numeric_cols) >= 2 ? numeric_cols[2] : numeric_cols[1]
        limit = min(size(data.X, 1), 300)
        scatter = [
            Dict(
                "x" => Float64(data.X[i, Symbol(xcol)]),
                "y" => Float64(data.X[i, Symbol(ycol)]),
                "target" => display_label(dataset_id, data.y[i]),
            )
            for i in 1:limit
        ]
    end

    target_payload = spec.task == :regression ?
        Dict(
            "target_summary" => numeric_summary(data.y),
            "target_histogram" => histogram(data.y),
            "target_distribution" => Dict(),
        ) :
        Dict(
            "target_summary" => Dict(),
            "target_histogram" => Dict(),
            "target_distribution" => display_counts(dataset_id, data.y),
        )

    col_names = names(data.X)
    n_sample = min(8, size(data.X, 1))
    sample_rows = [
        merge(
            Dict(string(c) => data.X[i, c] for c in col_names),
            Dict("target" => display_label(dataset_id, data.y[i])),
        )
        for i in 1:n_sample
    ]
    tabular_samples = Dict(
        "columns" => [string.(col_names)..., "target"],
        "rows" => sample_rows,
    )

    payload = Dict(
        "dataset_id" => dataset_id,
        "dataset" => spec.label,
        "task" => string(spec.task),
        "description" => get(DATASET_DESCRIPTIONS, dataset_id, ""),
        "n_samples" => size(data.X, 1),
        "n_features" => size(data.X, 2),
        "feature_schema" => schema,
        "numeric" => numeric,
        "categorical" => categorical,
        "correlation" => isempty(numeric_cols) ? Dict("columns" => [], "matrix" => []) : correlation_matrix(data.X, numeric_cols),
        "scatter" => Dict(
            "x_feature" => isempty(numeric_cols) ? "" : numeric_cols[1],
            "y_feature" => length(numeric_cols) >= 2 ? numeric_cols[2] : (isempty(numeric_cols) ? "" : numeric_cols[1]),
            "points" => scatter,
        ),
        "samples" => tabular_samples,
    )
    merge!(payload, target_payload)
    return payload
end

function handle_home(req)
    return html_response(read(joinpath(@__DIR__, "views/demo_index.html"), String))
end

function handle_inference_page(req)
    return html_response(read(joinpath(@__DIR__, "views/demo_inference.html"), String))
end

function handle_visualization_page(req)
    return html_response(read(joinpath(@__DIR__, "views/demo_visualization.html"), String))
end

function handle_api_catalog(req)
    return json_response(catalog_payload())
end

function handle_api_train(req)
    try
        body = isempty(req.body) ? Dict{String, Any}() : JSON.parse(String(req.body))
        dataset_id = get(body, "dataset", current_dataset_id == "" ? "iris" : current_dataset_id)
        dataset_task = DATASETS[dataset_id].task
        model_id = get(body, "model", default_model_id(dataset_task))
        metrics = train_demo_model(dataset_id, model_id)
        return json_response(Dict("ok" => true, "metrics" => metrics, "model" => model_info_payload()))
    catch e
        return json_response(Dict("ok" => false, "error" => sprint(showerror, e)), status=400)
    end
end

function model_info_payload()
    current_model === nothing && return Dict("error" => "No model loaded")

    info = Dict(
        "dataset_id" => current_dataset_id,
        "dataset" => current_dataset_name,
        "model_id" => current_model_id,
        "model" => current_model_name,
        "model_type" => string(typeof(current_model)),
        "task" => string(current_task),
        "is_trained" => current_model.is_trained,
        "metrics" => current_metrics,
    )

    if current_task == :timeseries
        info["features"] = []
        info["classes"] = []
        info["n_samples"] = length(current_data.y)
    else
        info["features"] = names(current_data.X)
        info["classes"] = current_task == :classification ? [display_label(current_dataset_id, c) for c in unique(current_data.y)] : []
        info["n_samples"] = size(current_data.X, 1)
    end

    return info
end

function handle_api_model_info(req)
    return json_response(model_info_payload())
end

function prediction_dataframe(input_data)
    cols = Dict{Symbol, Any}()
    for item in feature_schema()
        name = item["name"]
        if item["kind"] == "number"
            cols[Symbol(name)] = [Float64(get(input_data, name, item["mean"]))]
        else
            cols[Symbol(name)] = [get(input_data, name, item["default"])]
        end
    end
    return DataFrame(cols)
end

function handle_api_predict(req)
    current_model === nothing && return json_response(Dict("error" => "No model loaded"))

    try
        input_data = JSON.parse(String(req.body))

        if current_task == :timeseries
            steps = Int(get(input_data, "steps", 12))
            forecast = predict(current_model, steps)
            return json_response(Dict(
                "task" => "timeseries",
                "steps" => steps,
                "forecast" => forecast,
            ))
        end

        test_df = prediction_dataframe(input_data)
        prediction = predict(current_model, test_df)[1]
        proba = nothing

        if current_task == :classification
            try
                proba_dist = predict_proba(current_model, test_df)[1]
                proba = Dict(display_label(current_dataset_id, label) => MLJ.pdf(proba_dist, label) for label in unique(current_data.y))
            catch
                proba = nothing
            end
        end

        return json_response(Dict(
            "task" => string(current_task),
            "prediction" => current_task == :classification ? display_label(current_dataset_id, prediction) : (prediction isa Number ? prediction : string(prediction)),
            "probabilities" => proba,
            "raw_prediction" => string(prediction),
            "input" => input_data,
        ))
    catch e
        return json_response(Dict(
            "error" => string(e),
            "stacktrace" => sprint(showerror, e, catch_backtrace()),
        ))
    end
end

function handle_api_dataset_stats(req)
    current_data === nothing && return json_response(Dict("error" => "No dataset loaded"))

    if current_task == :timeseries
        y = current_data.y
        return json_response(Dict(
            "dataset" => current_dataset_name,
            "task" => string(current_task),
            "n_samples" => length(y),
            "n_features" => 0,
            "series_stats" => Dict(
                "min" => minimum(y),
                "max" => maximum(y),
                "mean" => mean(y),
                "std" => std(y),
            ),
            "feature_stats" => Dict(),
            "target_distribution" => Dict(),
        ))
    end

    stats = Dict()
    for item in feature_schema()
        if item["kind"] == "number"
            stats[item["name"]] = item
        else
            stats[item["name"]] = Dict(
                "kind" => "category",
                "values" => item["values"],
                "counts" => Dict(v => count(==(v), string.(current_data.X[!, Symbol(item["name"])])) for v in item["values"]),
            )
        end
    end

    return json_response(Dict(
        "dataset" => current_dataset_name,
        "task" => string(current_task),
        "n_samples" => size(current_data.X, 1),
        "n_features" => size(current_data.X, 2),
        "feature_schema" => feature_schema(),
        "feature_stats" => stats,
        "target_distribution" => current_task == :classification ? display_counts(current_dataset_id, current_data.y) : Dict(
            string(c) => count(==(c), current_data.y)
            for c in unique(current_data.y)
        ),
    ))
end

function handle_api_visualization_data(req, query)
    try
        dataset_id = get(query, "dataset", current_dataset_id == "" ? "iris" : current_dataset_id)
        return json_response(visualization_payload(dataset_id))
    catch e
        return json_response(Dict("error" => sprint(showerror, e)), status=400)
    end
end

function router(req)
    path, query = request_path_and_query(req.target)

    try
        if path == "/"
            return handle_home(req)
        elseif path == "/inference"
            return handle_inference_page(req)
        elseif path == "/visualization"
            return handle_visualization_page(req)
        elseif path == "/api/catalog"
            return handle_api_catalog(req)
        elseif path == "/api/train" && req.method == "POST"
            return handle_api_train(req)
        elseif path == "/api/model-info"
            return handle_api_model_info(req)
        elseif path == "/api/predict" && req.method == "POST"
            return handle_api_predict(req)
        elseif path == "/api/dataset-stats"
            return handle_api_dataset_stats(req)
        elseif path == "/api/visualization-data"
            return handle_api_visualization_data(req, query)
        else
            return HTTP.Response(404, "Not Found")
        end
    catch e
        println("Error handling request: ", e)
        return HTTP.Response(500, "Internal Server Error: $(string(e))")
    end
end

function main(port=8000)
    println("=" ^ 60)
    println("DSToolkit Demo Server")
    println("=" ^ 60)

    train_demo_model()

    println("\nStarting server on http://localhost:$port")
    println("   Home: http://localhost:$port/")
    println("   Inference: http://localhost:$port/inference")
    println("   Visualization: http://localhost:$port/visualization")
    println("\nPress Ctrl+C to stop the server")
    println("=" ^ 60)

    HTTP.serve(router, "127.0.0.1", port)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
