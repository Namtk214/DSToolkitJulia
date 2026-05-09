# Distribution Visualization
# Requires Plots.jl to be loaded by the user: using Plots

# Helper function to check if Plots is loaded
function _check_plots()
    if !isdefined(Main, :Plots)
        error("Plots.jl is required for visualization. First install it with: using Pkg; Pkg.add(\"Plots\"), then load it with: using Plots")
    end
end

"""
    plot_histogram(data::ToolkitData, col::Symbol)

Plot histogram for a specific column.

# Requirements
Requires Plots.jl to be loaded: `using Plots`

# Example
```julia
using DSToolkit
using Plots

data = load_iris()
plot_histogram(data, :sepal_length)
```
"""
function plot_histogram(data::ToolkitData, col::Symbol)
    _check_plots()
    data.X === nothing && error("Cannot plot histogram for time series data")

    if col in Symbol.(names(data.X))
        vals = data.X[!, col]
        return Main.Plots.histogram(vals, xlabel=string(col), ylabel="Frequency",
                 title="Distribution of $(col)", legend=false,
                 color=:steelblue, alpha=0.7)
    else
        error("Column $col not found in data")
    end
end

"""
    plot_boxplot(data::ToolkitData, col::Symbol)

Plot boxplot for a specific column.

# Requirements
Requires Plots.jl to be loaded: `using Plots`
"""
function plot_boxplot(data::ToolkitData, col::Symbol)
    _check_plots()
    data.X === nothing && error("Cannot plot boxplot for time series data")

    if col in Symbol.(names(data.X))
        vals = data.X[!, col]
        return Main.Plots.boxplot([string(col)], [vals], ylabel=string(col),
               title="Boxplot of $(col)", legend=false,
               color=:steelblue, alpha=0.7)
    else
        error("Column $col not found in data")
    end
end

"""
    plot_target_distribution(data::ToolkitData)

Plot distribution of the target variable.

# Requirements
Requires Plots.jl to be loaded: `using Plots`
"""
function plot_target_distribution(data::ToolkitData)
    _check_plots()

    if data.task == :classification
        # Bar chart for classification
        counts = Dict(v => count(==(v), data.y) for v in unique(data.y))
        labels = collect(keys(counts))
        values = [counts[l] for l in labels]
        return Main.Plots.bar(string.(labels), values, xlabel="Class", ylabel="Count",
           title="Target Distribution", legend=false,
           color=:steelblue, alpha=0.7)
    else
        # Histogram for regression/time series
        return Main.Plots.histogram(data.y, xlabel="Target Value", ylabel="Frequency",
                 title="Target Distribution", legend=false,
                 color=:steelblue, alpha=0.7)
    end
end
