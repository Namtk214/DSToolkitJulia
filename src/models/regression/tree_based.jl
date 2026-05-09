# Tree-Based Regression Models
# Includes: DecisionTreeReg, RandomForestReg, XGBoostReg

function fit!(model::DecisionTreeReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load DecisionTreeRegressor pkg=DecisionTree verbosity=0
    pipe = _continuous_encoder() |> ModelType(max_depth=model.max_depth)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Decision Tree Regressor trained (max_depth=$(model.max_depth))"
    return model
end

function fit!(model::RandomForestReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load RandomForestRegressor pkg=DecisionTree verbosity=0
    pipe = _continuous_encoder() |> ModelType(n_trees=model.n_trees)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Random Forest Regressor trained (n_trees=$(model.n_trees))"
    return model
end

function fit!(model::XGBoostReg, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load XGBoostRegressor pkg=XGBoost verbosity=0
    pipe = _continuous_encoder() |> ModelType(num_round=model.num_round, max_depth=model.max_depth)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ XGBoost Regressor trained (rounds=$(model.num_round), depth=$(model.max_depth))"
    return model
end
