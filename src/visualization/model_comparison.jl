# Model Comparison Visualization
# Requires Plots.jl to be loaded by the user

"""
    plot_comparison_results(result::ComparisonResult)

Plot bar chart comparing all models by their primary metric.

# Requirements
Requires Plots.jl to be loaded: `using Plots`
"""
function plot_comparison_results(result::ComparisonResult)
    _check_plots()

    df = result.results
    model_names = df[!, :Model]

    # Choose primary metric based on task
    if result.task == :regression
        metric = :RMSE
        title_str = "Model Comparison (RMSE - lower is better)"
    elseif result.task == :classification
        metric = :Accuracy
        title_str = "Model Comparison (Accuracy - higher is better)"
    else  # timeseries
        metric = :RMSE
        title_str = "Model Comparison (RMSE - lower is better)"
    end

    values = df[!, metric]

    # Highlight best model
    colors = [name == result.best_model_name ? :gold : :steelblue for name in model_names]

    return Main.Plots.bar(string.(model_names), values,
           xlabel="Model", ylabel=string(metric),
           title=title_str,
           legend=false, color=colors, alpha=0.7,
           xrotation=45, size=(1000, 600))
end

"""
    plot_metric_comparison(result::ComparisonResult, metric::Symbol)

Plot bar chart comparing all models by a specific metric.

# Requirements
Requires Plots.jl to be loaded: `using Plots`
"""
function plot_metric_comparison(result::ComparisonResult, metric::Symbol)
    _check_plots()

    df = result.results

    if !(metric in Symbol.(names(df)))
        error("Metric $metric not found in results")
    end

    model_names = df[!, :Model]
    values = df[!, metric]

    # Highlight best model
    colors = [name == result.best_model_name ? :gold : :steelblue for name in model_names]

    return Main.Plots.bar(string.(model_names), values,
           xlabel="Model", ylabel=string(metric),
           title="Model Comparison ($metric)",
           legend=false, color=colors, alpha=0.7,
           xrotation=45, size=(1000, 600))
end
