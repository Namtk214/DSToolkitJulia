
"""
    ingest_data(source; target=nothing, task=nothing)

Ingest clean data from various formats and normalize to `ToolkitData`.

# Supported input formats
- `DataFrame` with a target column: `ingest_data(df; target=:y)`
- Separate `X` and `y`: `ingest_data(X, y)`
- `Matrix` + `Vector`: `ingest_data(matrix, vec)`
- CSV file path: `ingest_data("data.csv"; target=:y)`
- JLD2 file path: `ingest_data("data.jld2")`
- `Dict` or `NamedTuple`: `ingest_data(dict; target=:y)`
- Univariate time series: `ingest_data(vec; task=:timeseries)`

# Task auto-detection
- String/Categorical target or ≤20 unique values → `:classification`
- Continuous numeric target → `:regression`
- Single vector, no features → `:timeseries`
- Override with `task=:classification`, `:regression`, or `:timeseries`
"""
function ingest_data end

# --- DataFrame with target column ---
function ingest_data(df::DataFrame; target::Union{Symbol,String,Nothing}=nothing,
                     task::Union{Symbol,Nothing}=nothing)
    ncol(df) < 2 && error("DataFrame must have at least 2 columns (features + target).")

    target_sym = if target === nothing
        Symbol(names(df)[end])
    else
        Symbol(target)
    end

    target_sym in Symbol.(names(df)) || error("Target column ':$target_sym' not found in DataFrame.")

    y = df[!, target_sym]
    X = select(df, Not(target_sym))

    detected = task !== nothing ? task : detect_task(y)

    if detected == :classification
        y = categorical(y)
    end

    validate_data(X, y)
    return ToolkitData(X, y, detected)
end

# --- Separate X (DataFrame) and y ---
function ingest_data(X::DataFrame, y::AbstractVector;
                     task::Union{Symbol,Nothing}=nothing)
    detected = task !== nothing ? task : detect_task(y)
    if detected == :classification
        y = categorical(y)
    end
    validate_data(X, y)
    return ToolkitData(X, y, detected)
end

# --- Separate X (Matrix) and y ---
function ingest_data(X::AbstractMatrix, y::AbstractVector;
                     task::Union{Symbol,Nothing}=nothing)
    col_names = [Symbol("x$i") for i in 1:size(X, 2)]
    df = DataFrame(X, col_names)
    return ingest_data(df, y; task=task)
end

# --- File path (CSV or JLD2) ---
function ingest_data(path::String; target::Union{Symbol,String,Nothing}=nothing,
                     task::Union{Symbol,Nothing}=nothing)
    isfile(path) || error("File not found: $path")

    if endswith(lowercase(path), ".csv")
        df = CSV.read(path, DataFrame)
        return ingest_data(df; target=target, task=task)
    elseif endswith(lowercase(path), ".jld2")
        data = JLD2.load(path)
        # Support {"X" => ..., "y" => ...} or {:X => ..., :y => ...}
        X_key = haskey(data, "X") ? "X" : haskey(data, :X) ? :X : nothing
        y_key = haskey(data, "y") ? "y" : haskey(data, :y) ? :y : nothing

        if X_key !== nothing && y_key !== nothing
            X_raw = data[X_key]
            X_df = X_raw isa DataFrame ? X_raw : DataFrame(X_raw, :auto)
            return ingest_data(X_df, data[y_key]; task=task)
        elseif haskey(data, "data") || haskey(data, :data)
            d_key = haskey(data, "data") ? "data" : :data
            return ingest_data(data[d_key]; target=target, task=task)
        else
            error("JLD2 file must contain keys 'X'/'y' or 'data'. Found: $(keys(data))")
        end
    else
        error("Unsupported file format: $(splitext(path)[2]). Supported: .csv, .jld2")
    end
end

# --- Dict ---
function ingest_data(d::AbstractDict; target::Union{Symbol,String,Nothing}=nothing,
                     task::Union{Symbol,Nothing}=nothing)
    # Try X/y keys (string or symbol)
    X_key = haskey(d, "X") ? "X" : haskey(d, :X) ? :X : nothing
    y_key = haskey(d, "y") ? "y" : haskey(d, :y) ? :y : nothing

    if X_key !== nothing && y_key !== nothing
        X_raw = d[X_key]
        X_df = X_raw isa DataFrame ? X_raw : DataFrame(X_raw, :auto)
        return ingest_data(X_df, d[y_key]; task=task)
    else
        # Try building a DataFrame from the dict
        df = DataFrame(Dict(Symbol(k) => v for (k, v) in d))
        return ingest_data(df; target=target, task=task)
    end
end

# --- NamedTuple ---
function ingest_data(nt::NamedTuple; target::Union{Symbol,String,Nothing}=nothing,
                     task::Union{Symbol,Nothing}=nothing)
    if haskey(nt, :X) && haskey(nt, :y)
        X_raw = nt.X
        X_df = X_raw isa DataFrame ? X_raw : DataFrame(X_raw, :auto)
        return ingest_data(X_df, nt.y; task=task)
    else
        df = DataFrame(; nt...)
        return ingest_data(df; target=target, task=task)
    end
end

# --- Univariate time series (single vector) ---
function ingest_data(y::AbstractVector{<:Number}; task::Symbol=:timeseries)
    task == :timeseries || error("Single numeric vector input requires task=:timeseries.")
    validate_timeseries(y)
    return ToolkitData(nothing, collect(Float64, y), :timeseries)
end

# Task Detection

"""
    detect_task(y) → Symbol

Auto-detect whether target vector `y` represents a `:classification` or `:regression` task.
"""
function detect_task(y::AbstractVector)
    # Categorical or string types → classification
    if eltype(y) <: Union{AbstractString, CategoricalValue}
        return :classification
    end

    unique_count = length(unique(y))
    total_count = length(y)

    # Integer-like with few unique values → classification
    if eltype(y) <: Integer && unique_count <= 20
        return :classification
    end

    # Float but very few unique values → likely classification
    if unique_count <= min(20, max(2, total_count ÷ 10))
        return :classification
    end

    return :regression
end
