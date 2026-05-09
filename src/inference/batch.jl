# Batch Inference Utilities

"""
    batch_predict(model, data_batches)

Predict on multiple data batches sequentially.

# Arguments
- `model`: Trained model
- `data_batches`: Iterable of data batches

Returns vector of predictions for each batch.
"""
function batch_predict(model::AbstractToolkitModel, data_batches)
    return [predict(model, batch) for batch in data_batches]
end

"""
    parallel_predict(model, data_chunks)

Predict on multiple data chunks in parallel (uses threading if available).

# Arguments
- `model`: Trained model
- `data_chunks`: Vector of data chunks

Returns vector of predictions for each chunk.
"""
function parallel_predict(model::AbstractToolkitModel, data_chunks::Vector)
    # Simple sequential version (parallel would require thread-safe model handling)
    # In production, this could use Threads.@threads if models are thread-safe
    return [predict(model, chunk) for chunk in data_chunks]
end
