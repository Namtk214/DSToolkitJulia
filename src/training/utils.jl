# Training Utilities for DSToolkit

"""
    _check_before_fit(X, y)

Pre-fit validation to ensure data is not empty and dimensions match.
"""
function _check_before_fit(X::DataFrame, y::AbstractVector)
    nrow(X) == 0 && error("Cannot train on empty data.")
    length(y) == 0 && error("Cannot train on empty target.")
    nrow(X) != length(y) && error("Row mismatch: X=$(nrow(X)), y=$(length(y)).")
end

"""
    _prepare_tabular_features(X)

Coerce non-numeric tabular columns to finite categorical scitypes so MLJ
encoders can turn them into numeric features for backends that require
`Matrix{<:Real}`.
"""
function _categorical_feature_levels(X::DataFrame)
    levels = Dict{Symbol, Vector{String}}()
    for col in names(X)
        vals = X[!, col]
        if !(eltype(vals) <: Number)
            levels[Symbol(col)] = unique(string.(vals))
        end
    end
    return levels
end

function _prepare_tabular_features(X::DataFrame, levels::Dict{Symbol, Vector{String}}=Dict{Symbol, Vector{String}}())
    X_prepared = copy(X)
    coercions = Pair[]

    for col in names(X_prepared)
        vals = X_prepared[!, col]
        if !(eltype(vals) <: Number)
            col_sym = Symbol(col)
            X_prepared[!, col] = categorical(string.(vals))
            if haskey(levels, col_sym)
                CategoricalArrays.levels!(X_prepared[!, col], levels[col_sym])
            end
            push!(coercions, col_sym => MLJ.Multiclass)
        end
    end

    return isempty(coercions) ? X_prepared : MLJ.coerce(X_prepared, coercions...)
end

function _fit_tabular_features!(model::AbstractToolkitModel, X::DataFrame)
    levels = _categorical_feature_levels(X)
    if :_feature_levels in fieldnames(typeof(model))
        model._feature_levels = levels
    end
    return _prepare_tabular_features(X, levels)
end

_continuous_encoder() = MLJ.ContinuousEncoder(drop_last=true)

"""
    make_timeseries_windows(y; seq_len=12, horizon=1)

Convert a univariate series into supervised sequence tensors for deep time
series models.

Returns `X, target` where:
- `X` has shape `(1, seq_len, samples)`
- `target` has shape `(1, samples)`
"""
function make_timeseries_windows(y::AbstractVector; seq_len::Int=12, horizon::Int=1)
    seq_len > 0 || error("seq_len must be positive.")
    horizon > 0 || error("horizon must be positive.")

    values = Float32.(y)
    n_samples = length(values) - seq_len - horizon + 1
    n_samples > 0 || error("Time series length $(length(values)) is too short for seq_len=$seq_len and horizon=$horizon.")

    X = Array{Float32}(undef, 1, seq_len, n_samples)
    target = Array{Float32}(undef, 1, n_samples)

    for i in 1:n_samples
        X[1, :, i] = values[i:i+seq_len-1]
        target[1, i] = values[i+seq_len+horizon-1]
    end

    return X, target
end
