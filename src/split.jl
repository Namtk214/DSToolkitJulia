
"""
    train_test_split(data::ToolkitData; ratio=0.8)

Split `ToolkitData` into train/test sets. For tabular data, rows are randomly
shuffled. For statistical time series, the split is sequential (preserving
temporal order).

Returns different tuple shapes depending on task type:
- Tabular: `(X_train, X_test, y_train, y_test)`
- Time series: `(y_train, y_test)`
"""
function train_test_split(data::ToolkitData; ratio=training_config(:split).train_ratio)
    if data.task in (:regression, :classification)
        return train_test_split(data.X, data.y; ratio=ratio)
    else
        return train_test_split(data.y; ratio=ratio)
    end
end

"""
    train_test_split(X::DataFrame, y::AbstractVector; ratio=0.8)

Random 80/20 split for tabular data.
"""
function train_test_split(X::DataFrame, y::AbstractVector; ratio=training_config(:split).train_ratio)
    n = nrow(X)
    n < 2 && error("Need at least 2 samples to split.")
    idx = randperm(n)
    split_pt = floor(Int, n * ratio)
    split_pt = clamp(split_pt, 1, n - 1)  # Ensure both sets are non-empty
    train_idx = idx[1:split_pt]
    test_idx = idx[split_pt+1:end]
    return X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
end

"""
    train_test_split(X::AbstractArray{T,3}, y::AbstractMatrix; ratio=0.8)

Random 80/20 split for deep time series data.
`X` shape: `(features, seq_len, samples)`, `y` shape: `(output_dim, samples)`.
Shuffles along the samples (3rd) dimension.
"""
function train_test_split(X::AbstractArray{T,3}, y::AbstractMatrix; ratio=training_config(:split).train_ratio) where T
    n = size(X, 3)
    n < 2 && error("Need at least 2 samples to split.")
    idx = randperm(n)
    split_pt = floor(Int, n * ratio)
    split_pt = clamp(split_pt, 1, n - 1)
    train_idx = idx[1:split_pt]
    test_idx = idx[split_pt+1:end]
    return X[:, :, train_idx], X[:, :, test_idx], y[:, train_idx], y[:, test_idx]
end

"""
    train_test_split(y::AbstractVector; ratio=0.8)

Sequential split for univariate time series. No shuffling — temporal order is
preserved, which is critical for time series validity.
"""
function train_test_split(y::AbstractVector; ratio=training_config(:split).train_ratio)
    n = length(y)
    n < 2 && error("Need at least 2 points to split.")
    split_pt = floor(Int, n * ratio)
    split_pt = clamp(split_pt, 1, n - 1)
    return y[1:split_pt], y[split_pt+1:end]
end
