# Time Series Visualization
# Requires Plots.jl to be loaded by the user

"""
    plot_timeseries(y_train::AbstractVector, y_test::AbstractVector, predictions::AbstractVector; title="Time Series Forecast")

Plot time series with train, test, and predictions.

# Requirements
Requires Plots.jl to be loaded: `using Plots`
"""
function plot_timeseries(y_train::AbstractVector, y_test::AbstractVector,
                        predictions::AbstractVector; title::String="Time Series Forecast")
    _check_plots()

    n_train = length(y_train)
    n_test = length(y_test)

    # Time indices
    train_idx = 1:n_train
    test_idx = (n_train+1):(n_train+n_test)

    p = Main.Plots.plot(train_idx, y_train, label="Train", color=:blue, linewidth=2)
    Main.Plots.plot!(p, test_idx, y_test, label="Test (Actual)", color=:green, linewidth=2)
    Main.Plots.plot!(p, test_idx, predictions, label="Predictions", color=:red,
         linewidth=2, linestyle=:dash)
    Main.Plots.xlabel!(p, "Time")
    Main.Plots.ylabel!(p, "Value")
    Main.Plots.title!(p, title)
    Main.Plots.legend!(p, :topright)
    return p
end

"""
    plot_forecast(model::StatTimeSeriesModel, y_train::AbstractVector, horizon::Int)

Plot historical data and forecast for a statistical time series model.

# Requirements
Requires Plots.jl to be loaded: `using Plots`
"""
function plot_forecast(model::StatTimeSeriesModel, y_train::AbstractVector, horizon::Int)
    _check_plots()

    !model.is_trained && error("Model must be trained before forecasting")

    # Generate forecast
    forecast = predict(model, horizon)

    n_train = length(y_train)
    train_idx = 1:n_train
    forecast_idx = (n_train+1):(n_train+horizon)

    p = Main.Plots.plot(train_idx, y_train, label="Historical", color=:blue, linewidth=2)
    Main.Plots.plot!(p, forecast_idx, forecast, label="Forecast",
         color=:red, linewidth=2, linestyle=:dash)
    Main.Plots.xlabel!(p, "Time")
    Main.Plots.ylabel!(p, "Value")
    Main.Plots.title!(p, "Time Series Forecast ($(horizon) steps ahead)")
    Main.Plots.legend!(p, :topright)
    return p
end
