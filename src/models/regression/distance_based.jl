# Distance-Based Regression Models
# Includes: KNNReg, SVMReg

function fit!(model::KNNReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load KNNRegressor pkg=NearestNeighborModels verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType(K=model.K)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ KNN Regressor trained (K=$(model.K))"
    return model
end

function fit!(model::SVMReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load EpsilonSVR pkg=LIBSVM verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType()
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ SVM Regressor trained"
    return model
end
