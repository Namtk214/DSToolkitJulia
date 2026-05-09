# Result Display Utilities

"""
    _display_results(df, task, best_name)

Pretty-print comparison results table with best model highlighted.
"""
function _display_results(df::DataFrame, task::Symbol, best_name::String)
    println()

    # Mark the best model
    display_df = copy(df)
    display_df.Model = [n == best_name ? "⭐ $n" : n for n in df.Model]

    # Round numeric columns for display
    for col in names(display_df)
        if eltype(display_df[!, col]) <: Float64
            display_df[!, col] = round.(display_df[!, col]; digits=4)
        end
    end

    task_label = if task == :regression
        "REGRESSION"
    elseif task == :classification
        "CLASSIFICATION"
    elseif task == :timeseries
        "STATISTICAL TIME SERIES"
    else
        "DEEP LEARNING TIME SERIES"
    end

    header = "📊 $task_label — Model Comparison Results"

    println("┌" * "─"^(length(header)+2) * "┐")
    println("│ $header │")
    println("└" * "─"^(length(header)+2) * "┘")

    pretty_table(display_df)

    println()
    primary_metric = if task == :regression
        "RMSE: $(round(df[df.Model .== best_name, :RMSE][1]; digits=4))"
    elseif task == :classification
        "Accuracy: $(round(df[df.Model .== best_name, :Accuracy][1]; digits=4))"
    else
        "RMSE: $(round(df[df.Model .== best_name, :RMSE][1]; digits=4))"
    end
    println("  ⭐ Best Model: $best_name ($primary_metric)")
    println()
end
