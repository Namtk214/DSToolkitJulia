# High-Level Inference API

"""
    inference(model, data)

Alias for `predict`. High-level inference API for trained models.
"""
inference(model::AbstractToolkitModel, data) = predict(model, data)

"""
    inference_with_preprocessing(model, raw_data, preprocess_fn)

Perform inference with preprocessing pipeline.

# Arguments
- `model`: Trained model
- `raw_data`: Raw input data
- `preprocess_fn`: Function to preprocess data (e.g., `x -> standardize(x)`)

# Example
```julia
model = load_toolkit_model("model.jld2")
result = inference_with_preprocessing(model, raw_df, standardize)
```
"""
function inference_with_preprocessing(model::AbstractToolkitModel, raw_data,
                                     preprocess_fn::Function)
    processed_data = preprocess_fn(raw_data)
    return predict(model, processed_data)
end
