"""
    save_model(model::AbstractToolkitModel, filepath::String)

Save a trained model to disk as a `.jld2` file. The file stores the model struct
along with its fitted state.

# Flux models
For deep learning models, the Flux chain weights are saved separately via
`Flux.state()` for robust serialization.

# MLJ models
MLJ machines are saved directly. If this fails, hyperparameters and training
state are preserved so the model can be retrained.
"""
function save_model(model::AbstractToolkitModel, filepath::String)
    !model.is_trained && @warn "Saving an untrained model. It won't be able to predict after loading."

    path = endswith(filepath, ".jld2") ? filepath : filepath * ".jld2"

    if model isa DeepTimeSeriesModel
        _save_deep_ts(model, path)
    else
        try
            JLD2.jldsave(path; model=model)
        catch e
            @warn "Standard save failed, attempting fallback" exception=(e, catch_backtrace())
            _save_fallback(model, path)
        end
    end

    @info "💾 Model saved to: $path"
    return path
end

function _save_deep_ts(model::DeepTimeSeriesModel, path::String)
    chain_state = model._chain !== nothing ? Flux.state(model._chain) : nothing
    model_type = typeof(model)
    model_params = Dict(
        :input_dim  => model.input_dim,
        :hidden_dim => model.hidden_dim,
        :epochs     => model.epochs,
        :seq_len    => model.seq_len,
        :is_trained => model.is_trained,
        :train_data => model._train_data,
        :y_mean     => model._y_mean,
        :y_std      => model._y_std,
    )
    JLD2.jldsave(path;
        model_type = string(model_type),
        model_params = model_params,
        chain_state = chain_state
    )
end

function _save_fallback(model::AbstractToolkitModel, path::String)
    # Strip the internal machine/model and save metadata + hyperparams
    model_type = string(typeof(model))
    fields = Dict{Symbol,Any}()
    for fname in fieldnames(typeof(model))
        val = getfield(model, fname)
        if fname in (:machine, :_model, :_chain)
            continue  # Skip non-serializable internals
        end
        fields[fname] = val
    end
    JLD2.jldsave(path; model_type=model_type, fields=fields, fallback=true)
    @warn "Model saved in fallback mode. Internal state was not preserved — you may need to retrain."
end

"""
    load_toolkit_model(filepath::String) → AbstractToolkitModel

Load a previously saved model from a `.jld2` file. The returned model is ready
for `predict()` calls (if it was trained before saving).
"""
function load_toolkit_model(filepath::String)
    isfile(filepath) || error("File not found: $filepath")

    data = JLD2.load(filepath)

    # Check if this is a deep TS model saved with Flux.state
    if haskey(data, "model_type") && haskey(data, "chain_state")
        return _load_deep_ts(data)
    end

    # Check if this is a fallback save
    if haskey(data, "fallback") && data["fallback"] == true
        @warn "Loading a fallback-saved model. Internal fitted state was not preserved."
        return _load_fallback(data)
    end

    # Standard JLD2 load
    if haskey(data, "model")
        model = data["model"]
        @info "✅ Model loaded: $(typeof(model).name.name) (trained=$(model.is_trained))"
        return model
    end

    error("Unrecognized save format. Keys found: $(keys(data))")
end

function _load_deep_ts(data::Dict)
    type_str = data["model_type"]
    params = data["model_params"]
    chain_state = data["chain_state"]

    # Reconstruct model
    seq_len = get(params, :seq_len, 12)
    model = if contains(type_str, "RNNModel")
        RNNModel(params[:input_dim]; hidden_dim=params[:hidden_dim], epochs=params[:epochs], seq_len=seq_len)
    elseif contains(type_str, "LSTMModel")
        LSTMModel(params[:input_dim]; hidden_dim=params[:hidden_dim], epochs=params[:epochs], seq_len=seq_len)
    elseif contains(type_str, "GRUModel")
        GRUModel(params[:input_dim]; hidden_dim=params[:hidden_dim], epochs=params[:epochs], seq_len=seq_len)
    else
        error("Unknown deep TS model type: $type_str")
    end

    model._train_data = Float64.(get(params, :train_data, Float64[]))
    model._y_mean = Float64(get(params, :y_mean, 0.0))
    model._y_std = Float64(get(params, :y_std, 1.0))

    # Rebuild chain and load state
    if chain_state !== nothing
        rnn_type = if model isa RNNModel
            Flux.RNN
        elseif model isa LSTMModel
            Flux.LSTM
        else
            Flux.GRU
        end
        chain = SeqChain(rnn_type(model.input_dim => model.hidden_dim),
                         Flux.Dense(model.hidden_dim => 1))
        Flux.loadmodel!(chain, chain_state)
        model._chain = chain
        model.is_trained = params[:is_trained]
    end

    @info "✅ Deep TS model loaded: $(typeof(model).name.name) (trained=$(model.is_trained))"
    return model
end

function _load_fallback(data::Dict)
    type_str = data["model_type"]
    fields = data["fields"]

    @warn "Fallback-loaded model ($type_str) has no fitted state. Call fit!() to retrain."
    # Return a best-effort reconstruction
    # The user will need to retrain this model
    return fields
end
