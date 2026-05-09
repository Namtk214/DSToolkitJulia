# Data Containers for DSToolkit

"""
    ToolkitData

Standardized container returned by `ingest_data()`.

# Fields
- `X::Union{DataFrame, Nothing}`: Feature DataFrame (or `nothing` for univariate time series)
- `y::AbstractVector`: Target vector
- `task::Symbol`: Detected task type — `:regression`, `:classification`, or `:timeseries`

# Example
```julia
data = ingest_data(df; target=:price)
@assert data.task == :regression
@assert data.X isa DataFrame
@assert data.y isa AbstractVector
```
"""
struct ToolkitData
    X::Union{DataFrame, Nothing}
    y::AbstractVector
    task::Symbol
end

"""
    ComparisonResult

Returned by `auto_compare()`. Contains the full results table, the best model,
and all trained models for the user to pick from.

# Fields
- `task::Symbol`: Task type (`:regression`, `:classification`, `:timeseries`)
- `results::DataFrame`: Comparison table with all metrics
- `best_model::AbstractToolkitModel`: The top-performing trained model
- `best_model_name::String`: Name of the best model (e.g., "XGBoost")
- `all_models::Vector{Pair{String, AbstractToolkitModel}}`: All trained models with names

# Example
```julia
result = auto_compare(data)
println("Best model: ", result.best_model_name)
predictions = predict(result.best_model, X_test)
```
"""
struct ComparisonResult
    task::Symbol
    results::DataFrame
    best_model::AbstractToolkitModel
    best_model_name::String
    all_models::Vector{Pair{String, AbstractToolkitModel}}
end
