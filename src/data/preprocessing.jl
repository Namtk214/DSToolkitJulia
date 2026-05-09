# Data Preprocessing Utilities

"""
    impute_missing(data::ToolkitData; strategy=:mean)

Impute missing values in the dataset.

# Strategies
- `:mean` - Replace with column mean (numeric only)
- `:median` - Replace with column median (numeric only)
- `:mode` - Replace with most frequent value
- `:forward` - Forward fill (carry last valid value forward)

Returns a new `ToolkitData` with imputed values.
"""
function impute_missing(data::ToolkitData; strategy::Symbol=:mean)
    data.X === nothing && return data  # Skip for time series

    X_new = copy(data.X)

    for col in names(X_new)
        vals = X_new[!, col]
        missing_idx = ismissing.(vals)

        if !any(missing_idx)
            continue  # No missing values
        end

        if strategy == :mean && eltype(skipmissing(vals)) <: Number
            replacement = mean(skipmissing(vals))
            X_new[missing_idx, col] .= replacement
        elseif strategy == :median && eltype(skipmissing(vals)) <: Number
            replacement = median(skipmissing(vals))
            X_new[missing_idx, col] .= replacement
        elseif strategy == :mode
            # Most frequent value
            non_missing = collect(skipmissing(vals))
            unique_vals = unique(non_missing)
            counts = Dict(v => count(==(v), non_missing) for v in unique_vals)
            replacement = argmax(counts)
            X_new[missing_idx, col] .= replacement
        elseif strategy == :forward
            # Forward fill
            last_valid = nothing
            for i in 1:nrow(X_new)
                if !ismissing(vals[i])
                    last_valid = vals[i]
                elseif last_valid !== nothing
                    X_new[i, col] = last_valid
                end
            end
        else
            error("Unknown imputation strategy: $strategy")
        end
    end

    return ToolkitData(X_new, data.y, data.task)
end

"""
    standardize(data::ToolkitData)

Apply z-score standardization to numeric features: (x - μ) / σ

Returns a new `ToolkitData` with standardized features.
"""
function standardize(data::ToolkitData)
    data.X === nothing && return data  # Skip for time series

    X_new = copy(data.X)

    for col in names(X_new)
        vals = X_new[!, col]
        if eltype(vals) <: Number
            μ = mean(vals)
            σ = std(vals)
            if σ > 0
                X_new[!, col] = (vals .- μ) ./ σ
            end
        end
    end

    return ToolkitData(X_new, data.y, data.task)
end

"""
    normalize(data::ToolkitData)

Apply min-max normalization to numeric features: (x - min) / (max - min)

Scales all features to [0, 1] range.

Returns a new `ToolkitData` with normalized features.
"""
function normalize(data::ToolkitData)
    data.X === nothing && return data  # Skip for time series

    X_new = copy(data.X)

    for col in names(X_new)
        vals = X_new[!, col]
        if eltype(vals) <: Number
            min_val = minimum(vals)
            max_val = maximum(vals)
            range = max_val - min_val
            if range > 0
                X_new[!, col] = (vals .- min_val) ./ range
            end
        end
    end

    return ToolkitData(X_new, data.y, data.task)
end

"""
    one_hot_encode(data::ToolkitData, cols::Vector{Symbol})

One-hot encode categorical columns.

Each unique value in a column becomes a binary feature.

Returns a new `ToolkitData` with one-hot encoded features.
"""
function one_hot_encode(data::ToolkitData, cols::Vector{Symbol})
    data.X === nothing && return data

    X_new = copy(data.X)

    for col in cols
        if col in Symbol.(names(X_new))
            vals = X_new[!, col]
            unique_vals = unique(vals)

            # Create binary columns for each unique value
            for val in unique_vals
                new_col = Symbol("$(col)_$(val)")
                X_new[!, new_col] = Int.(vals .== val)
            end

            # Remove original column
            select!(X_new, Not(col))
        end
    end

    return ToolkitData(X_new, data.y, data.task)
end

"""
    label_encode(data::ToolkitData, cols::Vector{Symbol})

Label encode categorical columns to integers.

Maps unique values to integers 0, 1, 2, ...

Returns a new `ToolkitData` with label encoded features.
"""
function label_encode(data::ToolkitData, cols::Vector{Symbol})
    data.X === nothing && return data

    X_new = copy(data.X)

    for col in cols
        if col in Symbol.(names(X_new))
            vals = X_new[!, col]
            unique_vals = unique(vals)
            mapping = Dict(val => i-1 for (i, val) in enumerate(unique_vals))
            X_new[!, col] = [mapping[v] for v in vals]
        end
    end

    return ToolkitData(X_new, data.y, data.task)
end

"""
    add_polynomial_features(data::ToolkitData; degree=2)

Add polynomial features up to the specified degree.

For features x1, x2, adds: x1², x2², x1*x2 (degree=2)

Returns a new `ToolkitData` with polynomial features added.
"""
function add_polynomial_features(data::ToolkitData; degree::Int=2)
    data.X === nothing && return data
    degree < 2 && return data

    X_new = copy(data.X)
    numeric_cols = [col for col in names(X_new) if eltype(X_new[!, col]) <: Number]

    # Add squared terms
    if degree >= 2
        for col in numeric_cols
            new_col = Symbol("$(col)_squared")
            X_new[!, new_col] = X_new[!, col] .^ 2
        end
    end

    # Add cubic terms if needed
    if degree >= 3
        for col in numeric_cols
            new_col = Symbol("$(col)_cubed")
            X_new[!, new_col] = X_new[!, col] .^ 3
        end
    end

    return ToolkitData(X_new, data.y, data.task)
end

"""
    add_interaction_features(data::ToolkitData, col_pairs::Vector{Tuple{Symbol, Symbol}})

Add interaction features (products) for specified column pairs.

# Example
```julia
data = add_interaction_features(data, [(:feature1, :feature2), (:feature1, :feature3)])
```

Returns a new `ToolkitData` with interaction features added.
"""
function add_interaction_features(data::ToolkitData, col_pairs::Vector{Tuple{Symbol, Symbol}})
    data.X === nothing && return data

    X_new = copy(data.X)

    for (col1, col2) in col_pairs
        if col1 in Symbol.(names(X_new)) && col2 in Symbol.(names(X_new))
            vals1 = X_new[!, col1]
            vals2 = X_new[!, col2]

            if eltype(vals1) <: Number && eltype(vals2) <: Number
                new_col = Symbol("$(col1)_x_$(col2)")
                X_new[!, new_col] = vals1 .* vals2
            end
        end
    end

    return ToolkitData(X_new, data.y, data.task)
end
