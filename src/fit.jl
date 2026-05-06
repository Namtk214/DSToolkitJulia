# --- Pre-fit validation ---
function _check_before_fit(X::DataFrame, y::AbstractVector)
    nrow(X) == 0 && error("Cannot train on empty data.")
    length(y) == 0 && error("Cannot train on empty target.")
    nrow(X) != length(y) && error("Row mismatch: X=$(nrow(X)), y=$(length(y)).")
end

# Regression Models — Each gets its own fit! via dispatch

function fit!(model::LinearReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load LinearRegressor pkg=GLM verbosity=0
    pipe = MLJ.Standardizer() |> ModelType()
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Linear Regression trained"
    return model
end

function fit!(model::RidgeReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load RidgeRegressor pkg=MLJLinearModels verbosity=0
    pipe = MLJ.Standardizer() |> ModelType(lambda=model.lambda)
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Ridge Regression trained (λ=$(model.lambda))"
    return model
end

function fit!(model::LassoReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load LassoRegressor pkg=MLJLinearModels verbosity=0
    pipe = MLJ.Standardizer() |> ModelType(lambda=model.lambda)
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Lasso Regression trained (λ=$(model.lambda))"
    return model
end

function fit!(model::ElasticNetReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load ElasticNetRegressor pkg=MLJLinearModels verbosity=0
    pipe = MLJ.Standardizer() |> ModelType(lambda=model.lambda, gamma=model.alpha)
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Elastic Net trained (λ=$(model.lambda), α=$(model.alpha))"
    return model
end

function fit!(model::DecisionTreeReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load DecisionTreeRegressor pkg=DecisionTree verbosity=0
    mach = MLJ.machine(ModelType(max_depth=model.max_depth), X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Decision Tree Regressor trained (max_depth=$(model.max_depth))"
    return model
end

function fit!(model::RandomForestReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load RandomForestRegressor pkg=DecisionTree verbosity=0
    mach = MLJ.machine(ModelType(n_trees=model.n_trees), X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Random Forest Regressor trained (n_trees=$(model.n_trees))"
    return model
end

function fit!(model::XGBoostReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load XGBoostRegressor pkg=XGBoost verbosity=0
    mach = MLJ.machine(ModelType(num_round=model.num_round, max_depth=model.max_depth), X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ XGBoost Regressor trained (rounds=$(model.num_round), depth=$(model.max_depth))"
    return model
end

function fit!(model::KNNReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load KNNRegressor pkg=NearestNeighborModels verbosity=0
    pipe = MLJ.Standardizer() |> ModelType(K=model.K)
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ KNN Regressor trained (K=$(model.K))"
    return model
end

function fit!(model::SVMReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load EpsilonSVR pkg=LIBSVM verbosity=0
    pipe = MLJ.Standardizer() |> ModelType()
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ SVM Regressor trained"
    return model
end

# Classification Models — Each gets its own fit! via dispatch

function fit!(model::LogisticCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load LogisticClassifier pkg=MLJLinearModels verbosity=0
    pipe = MLJ.Standardizer() |> ModelType()
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Logistic Regression trained"
    return model
end

function fit!(model::DecisionTreeCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    mach = MLJ.machine(ModelType(max_depth=model.max_depth), X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Decision Tree Classifier trained (max_depth=$(model.max_depth))"
    return model
end

function fit!(model::RandomForestCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load RandomForestClassifier pkg=DecisionTree verbosity=0
    mach = MLJ.machine(ModelType(n_trees=model.n_trees), X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Random Forest Classifier trained (n_trees=$(model.n_trees))"
    return model
end

function fit!(model::AdaBoostCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
    mach = MLJ.machine(ModelType(n_iter=model.n_iter), X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ AdaBoost Classifier trained (n_iter=$(model.n_iter))"
    return model
end

function fit!(model::XGBoostCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load XGBoostClassifier pkg=XGBoost verbosity=0
    mach = MLJ.machine(ModelType(num_round=model.num_round, max_depth=model.max_depth), X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ XGBoost Classifier trained (rounds=$(model.num_round), depth=$(model.max_depth))"
    return model
end

function fit!(model::KNNCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
    pipe = MLJ.Standardizer() |> ModelType(K=model.K)
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ KNN Classifier trained (K=$(model.K))"
    return model
end

function fit!(model::SVMCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load SVC pkg=LIBSVM verbosity=0
    pipe = MLJ.Standardizer() |> ModelType()
    mach = MLJ.machine(pipe, X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ SVM Classifier trained"
    return model
end

function fit!(model::NaiveBayesCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    ModelType = MLJ.@load GaussianNBClassifier pkg=NaiveBayes verbosity=0
    mach = MLJ.machine(ModelType(), X, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Gaussian Naive Bayes trained"
    return model
end

# Statistical Time Series Models

function fit!(model::ARIMAModel, y::AbstractVector)
    validate_timeseries(y)
    y_float = Float64.(y)
    ssm_model = SSM.SARIMA(y_float; order=model.order)
    SSM.fit!(ssm_model)
    model._model = ssm_model
    model._train_data = y_float
    model.is_trained = true
    @info "✓ ARIMA$(model.order) trained on $(length(y)) points"
    return model
end

function fit!(model::ETSModel, y::AbstractVector)
    validate_timeseries(y)
    y_float = Float64.(y)
    ssm_model = SSM.ExponentialSmoothing(y_float)
    SSM.fit!(ssm_model)
    model._model = ssm_model
    model._train_data = y_float
    model.is_trained = true
    @info "✓ ETS trained on $(length(y)) points"
    return model
end

# Deep Learning Time Series Models

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

function _train_deep_ts!(model::DeepTimeSeriesModel, chain::SeqChain,
                         X::AbstractArray{Float32,3}, y::AbstractArray{Float32})
    opt_state = Flux.setup(Flux.Adam(0.01), chain)

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
