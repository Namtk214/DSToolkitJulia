# Linear Regression Models
# Includes: LinearReg, RidgeReg, LassoReg, ElasticNetReg

function fit!(model::LinearReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load LinearRegressor pkg=GLM verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType()
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Linear Regression trained"
    return model
end

function fit!(model::RidgeReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load RidgeRegressor pkg=MLJLinearModels verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType(lambda=model.lambda)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Ridge Regression trained (λ=$(model.lambda))"
    return model
end

function fit!(model::LassoReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load LassoRegressor pkg=MLJLinearModels verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType(lambda=model.lambda)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Lasso Regression trained (λ=$(model.lambda))"
    return model
end

function fit!(model::ElasticNetReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load ElasticNetRegressor pkg=MLJLinearModels verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType(lambda=model.lambda, gamma=model.alpha)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Elastic Net trained (λ=$(model.lambda), α=$(model.alpha))"
    return model
end
