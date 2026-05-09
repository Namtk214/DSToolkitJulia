# Correlation Visualization
# Requires Plots.jl to be loaded by the user

"""
    plot_correlation_heatmap(data::ToolkitData)

Plot correlation heatmap for all numeric features.

# Requirements
Requires Plots.jl to be loaded: `using Plots`
"""
function plot_correlation_heatmap(data::ToolkitData)
    _check_plots()
    data.X === nothing && error("Cannot plot correlation for time series data")

    # Get numeric columns only
    numeric_cols = [col for col in names(data.X) if eltype(data.X[!, col]) <: Number]

    if isempty(numeric_cols)
        error("No numeric columns found for correlation plot")
    end

    # Compute correlation matrix
    mat = Matrix{Float64}(data.X[!, numeric_cols])
    cor_mat = cor(mat, dims=1)

    # Plot heatmap
    return Main.Plots.heatmap(numeric_cols, numeric_cols, cor_mat,
           color=:RdBu, clims=(-1, 1),
           title="Feature Correlation Heatmap",
           xlabel="", ylabel="",
           xticks=(1:length(numeric_cols), string.(numeric_cols)),
           yticks=(1:length(numeric_cols), string.(numeric_cols)),
           xrotation=45, size=(800, 700))
end

"""
    plot_feature_vs_target(data::ToolkitData, feature::Symbol)

Plot scatter plot of a feature vs target (for regression).

# Requirements
Requires Plots.jl to be loaded: `using Plots`
"""
function plot_feature_vs_target(data::ToolkitData, feature::Symbol)
    _check_plots()
    data.X === nothing && error("Cannot plot feature vs target for time series data")
    data.task != :regression && error("This visualization is for regression tasks only")

    if feature in Symbol.(names(data.X))
        x_vals = data.X[!, feature]
        return Main.Plots.scatter(x_vals, data.y,
               xlabel=string(feature), ylabel="Target",
               title="$(feature) vs Target",
               legend=false, alpha=0.5,
               color=:steelblue, markersize=3)
    else
        error("Feature $feature not found in data")
    end
end
