# Logistic Regression Classification

function fit!(model::LogisticCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load LogisticClassifier pkg=MLJLinearModels verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType()
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Logistic Regression trained"
    return model
end
