# Data Validation Utilities

"""
    validate_data(X, y)

Validate tabular data before training.
Checks for:
- Empty data
- Row count mismatch
- NaN or Inf values in numeric columns
"""
function validate_data(X::DataFrame, y::AbstractVector)
    nrow(X) == 0 && error("Input DataFrame is empty.")
    length(y) == 0 && error("Target vector is empty.")
    nrow(X) != length(y) && error(
        "Row count mismatch: X has $(nrow(X)) rows, y has $(length(y)) elements.")

    for col in names(X)
        vals = X[!, col]
        if eltype(vals) <: Number
            if any(isnan, vals) || any(isinf, vals)
                error("Column '$col' contains NaN or Inf values. Please clean your data first.")
            end
        end
    end

    if eltype(y) <: Number
        if any(isnan, y) || any(isinf, y)
            error("Target vector contains NaN or Inf values.")
        end
    end
end

"""
    validate_timeseries(y)

Validate time series data before training.
Checks for:
- Minimum length (at least 10 points)
- NaN or Inf values
"""
function validate_timeseries(y::AbstractVector)
    length(y) < 10 && error("Time series too short ($(length(y)) points). Need at least 10.")
    if eltype(y) <: Number
        if any(isnan, y) || any(isinf, y)
            error("Time series contains NaN or Inf values.")
        end
    end
end
