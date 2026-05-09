# Deep Learning Time Series Models
# Includes: RNNModel, LSTMModel, GRUModel

"""
    fit!(model::DeepTimeSeriesModel, X, y)

Train a deep learning time series model.
- `X`: `(features, seq_len, samples)` Float32 array
- `y`: `(output_dim, samples)` Float32 array

Uses explicit gradient computation (modern Flux API) and proper sequence
handling via `SeqChain`.
"""
function fit!(model::RNNModel, X::AbstractArray{T,3}, y::AbstractArray) where T
    chain = SeqChain(Flux.RNN(model.input_dim => model.hidden_dim),
                     Flux.Dense(model.hidden_dim => size(y, 1)))
    _train_deep_ts!(model, chain, Float32.(X), Float32.(y))
    return model
end

function fit!(model::LSTMModel, X::AbstractArray{T,3}, y::AbstractArray) where T
    chain = SeqChain(Flux.LSTM(model.input_dim => model.hidden_dim),
                     Flux.Dense(model.hidden_dim => size(y, 1)))
    _train_deep_ts!(model, chain, Float32.(X), Float32.(y))
    return model
end

function fit!(model::GRUModel, X::AbstractArray{T,3}, y::AbstractArray) where T
    chain = SeqChain(Flux.GRU(model.input_dim => model.hidden_dim),
                     Flux.Dense(model.hidden_dim => size(y, 1)))
    _train_deep_ts!(model, chain, Float32.(X), Float32.(y))
    return model
end

"""
    fit!(model::DeepTimeSeriesModel, y; seq_len=model.seq_len)

Train a deep time-series model directly on a univariate series by converting it
to one-step-ahead sliding windows. This is the path used by the demo datasets.
"""
function fit!(model::DeepTimeSeriesModel, y::AbstractVector; seq_len::Int=model.seq_len)
    length(y) > seq_len || error("Need more than seq_len=$seq_len points to train $(typeof(model).name.name).")

    y_float = Float64.(y)
    μ = mean(y_float)
    σ = std(y_float)
    σ = σ == 0.0 ? 1.0 : σ
    y_scaled = (y_float .- μ) ./ σ

    X, target = make_timeseries_windows(y_scaled; seq_len=seq_len)

    model.seq_len = seq_len
    model._train_data = y_float
    model._y_mean = μ
    model._y_std = σ

    fit!(model, X, target)
    return model
end

"""
    _train_deep_ts!(model, chain, X, y)

Internal training helper for deep learning time series models.
Uses Adam optimizer and MSE loss.
"""
function _train_deep_ts!(model::DeepTimeSeriesModel, chain::SeqChain,
                         X::AbstractArray{Float32,3}, y::AbstractArray{Float32})
    η = training_config(:model_defaults, :deep_ts).learning_rate
    opt_state = Flux.setup(Flux.Adam(η), chain)

    for epoch in 1:model.epochs
        loss_val, grads = Flux.withgradient(chain) do m
            ŷ = m(X)
            Flux.mse(ŷ, y)
        end
        Flux.update!(opt_state, chain, grads[1])

        if epoch == model.epochs
            @info "  Final Loss (Epoch $epoch/$(model.epochs)): $(round(loss_val; digits=6))"
        end
    end

    model._chain = chain
    model.is_trained = true
    name = typeof(model).name.name
    @info "✓ $name trained ($(model.epochs) epochs)"
end
