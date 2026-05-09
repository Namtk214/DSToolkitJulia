# Custom Flux Layers for Deep Learning Time Series

"""
    SeqChain

Custom Flux model that properly handles sequence input for RNN/LSTM/GRU.

Input shape: `(features, seq_len, batch)`.

Iterates timesteps through the recurrent layer, takes the final hidden state,
and passes it through a dense output layer.

# Fields
- `rnn_layer`: Recurrent layer (RNN, LSTM, or GRU)
- `dense_layer`: Dense output layer

# Example
```julia
model = SeqChain(
    Flux.LSTM(10 => 32),
    Flux.Dense(32 => 1)
)
X = rand(Float32, 10, 20, 16)  # (features, seq_len, batch)
output = model(X)  # (1, 16) - uses final hidden state
```
"""
struct SeqChain
    rnn_layer
    dense_layer
end

Flux.@layer SeqChain

function (m::SeqChain)(X::AbstractArray{T,3}) where T
    Flux.reset!(m.rnn_layer)
    seq_len = size(X, 2)
    h = nothing
    for t in 1:seq_len
        h = m.rnn_layer(X[:, t, :])
    end
    return m.dense_layer(h)
end
