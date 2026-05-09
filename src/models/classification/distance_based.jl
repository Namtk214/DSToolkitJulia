# Distance-Based Classification Models
# Includes: KNNCls, SVMCls, NaiveBayesCls

function fit!(model::KNNCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType(K=model.K)
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ KNN Classifier trained (K=$(model.K))"
    return model
end

function fit!(model::SVMCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load SVC pkg=LIBSVM verbosity=0
    pipe = _continuous_encoder() |> MLJ.Standardizer() |> ModelType()
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ SVM Classifier trained"
    return model
end

function fit!(model::NaiveBayesCls, X::DataFrame, y::AbstractVector)
    _check_before_fit(X, y)
    X_fit = _fit_tabular_features!(model, X)
    ModelType = MLJ.@load GaussianNBClassifier pkg=NaiveBayes verbosity=0
    pipe = _continuous_encoder() |> ModelType()
    mach = MLJ.machine(pipe, X_fit, y)
    MLJ.fit!(mach, verbosity=0)
    model.machine = mach
    model.is_trained = true
    @info "✓ Gaussian Naive Bayes trained"
    return model
end
