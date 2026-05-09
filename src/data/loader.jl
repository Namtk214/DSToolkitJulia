# Dataset Loaders for Sample Datasets

"""
    load_iris()

Load the Iris classification dataset (3 classes, 150 samples, 4 features).
Returns `ToolkitData` ready for training.
"""
function load_iris()
    path = joinpath(@__DIR__, "../../data/tabular/classification/iris.csv")
    df = CSV.read(path, DataFrame)
    return ingest_data(df; target=:species)
end

"""
    load_titanic()

Load the Titanic binary classification dataset (891 samples, 7 features).
Returns `ToolkitData` ready for training.
"""
function load_titanic()
    path = joinpath(@__DIR__, "../../data/tabular/classification/titanic.csv")
    df = CSV.read(path, DataFrame)
    return ingest_data(df; target=:survived)
end

"""
    load_wine_quality()

Load the Wine Quality multi-class classification dataset (1599 samples, 11 features).
Returns `ToolkitData` ready for training.
"""
function load_wine_quality()
    path = joinpath(@__DIR__, "../../data/tabular/classification/wine_quality.csv")
    df = CSV.read(path, DataFrame)
    return ingest_data(df; target=:quality)
end

"""
    load_housing()

Load the Housing regression dataset (506 samples, 13 features).
Returns `ToolkitData` ready for training.
"""
function load_housing()
    path = joinpath(@__DIR__, "../../data/tabular/regression/housing.csv")
    df = CSV.read(path, DataFrame)
    return ingest_data(df; target=:PRICE)
end

"""
    load_diabetes()

Load the Diabetes regression dataset (442 samples, 10 features).
Returns `ToolkitData` ready for training.
"""
function load_diabetes()
    path = joinpath(@__DIR__, "../../data/tabular/regression/diabetes.csv")
    df = CSV.read(path, DataFrame)
    return ingest_data(df; target=:target)
end

"""
    load_synthetic_reg()

Load the Synthetic Regression dataset (1000 samples, 5 features, clean linear relationship).
Returns `ToolkitData` ready for training.
"""
function load_synthetic_reg()
    path = joinpath(@__DIR__, "../../data/tabular/regression/synthetic_reg.csv")
    df = CSV.read(path, DataFrame)
    return ingest_data(df; target=:target)
end

"""
    load_airline_passengers()

Load the Airline Passengers time series dataset (144 months, seasonal pattern).
Returns `ToolkitData` ready for training.
"""
function load_airline_passengers()
    path = joinpath(@__DIR__, "../../data/timeseries/airline_passengers.csv")
    df = CSV.read(path, DataFrame)
    y = df.passengers
    return ingest_data(y; task=:timeseries)
end

"""
    load_stock_prices()

Load the Stock Prices time series dataset (252 days, financial time series).
Returns `ToolkitData` ready for training.
"""
function load_stock_prices()
    path = joinpath(@__DIR__, "../../data/timeseries/stock_prices.csv")
    df = CSV.read(path, DataFrame)
    y = df.price
    return ingest_data(y; task=:timeseries)
end

"""
    load_temperature()

Load the Temperature time series dataset (365 days, weather data with annual cycle).
Returns `ToolkitData` ready for training.
"""
function load_temperature()
    path = joinpath(@__DIR__, "../../data/timeseries/temperature.csv")
    df = CSV.read(path, DataFrame)
    y = df.temperature
    return ingest_data(y; task=:timeseries)
end

"""
    list_datasets()

List all available sample datasets. Returns a dictionary mapping task types to dataset names.
"""
function list_datasets()
    return Dict(
        :classification => ["iris", "titanic", "wine_quality"],
        :regression => ["housing", "diabetes", "synthetic_reg"],
        :timeseries => ["airline_passengers", "stock_prices", "temperature"]
    )
end
