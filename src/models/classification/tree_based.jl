# Tree-Based Classification Models
# Includes: DecisionTreeCls, RandomForestCls, AdaBoostCls, XGBoostCls

function fit!(model::DecisionTreeCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    pipe = _continuous_encoder() |> ModelType(max_depth=model.max_depth)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Decision Tree Classifier trained (max_depth=$(model.max_depth))"
    return model
end

function fit!(model::RandomForestCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load RandomForestClassifier pkg=DecisionTree verbosity=0
    pipe = _continuous_encoder() |> ModelType(n_trees=model.n_trees)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Random Forest Classifier trained (n_trees=$(model.n_trees))"
    return model
end

function fit!(model::AdaBoostCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load AdaBoostStumpClassifier pkg=DecisionTree verbosity=0
    pipe = _continuous_encoder() |> ModelType(n_iter=model.n_iter)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ AdaBoost Classifier trained (n_iter=$(model.n_iter))"
    return model
end

function fit!(model::XGBoostCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load XGBoostClassifier pkg=XGBoost verbosity=0
    pipe = _continuous_encoder() |> ModelType(num_round=model.num_round, max_depth=model.max_depth)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ XGBoost Classifier trained (rounds=$(model.num_round), depth=$(model.max_depth))"
    return model
end
