using Test
using DSToolkit
using DataFrames

@testset "Data Loader Tests" begin

    @testset "Classification Datasets" begin
        # Test Iris dataset
        @testset "Iris" begin
            data = load_iris()
            @test data.task == :classification
            @test size(data.X, 1) == 150  # 150 samples
            @test size(data.X, 2) == 4    # 4 features
            @test length(data.y) == 150
            @test length(unique(data.y)) == 3  # 3 classes
            @test all(x -> x in ["setosa", "versicolor", "virginica"], data.y)
        end

        # Test Titanic dataset
        @testset "Titanic" begin
            data = load_titanic()
            @test data.task == :classification
            @test size(data.X, 1) == 891
            @test length(data.y) == 891
            @test all(x -> x in [0, 1], data.y)  # Binary classification
        end

        # Test Wine Quality dataset
        @testset "Wine Quality" begin
            data = load_wine_quality()
            @test data.task == :classification
            @test size(data.X, 1) == 1599
            @test length(data.y) == 1599
            # Convert categorical to int for comparison
            y_vals = Int.(data.y)
            @test minimum(y_vals) >= 3
            @test maximum(y_vals) <= 8
        end
    end

    @testset "Regression Datasets" begin
        # Test Housing dataset
        @testset "Housing" begin
            data = load_housing()
            @test data.task == :regression
            @test size(data.X, 1) == 506
            @test size(data.X, 2) == 13
            @test length(data.y) == 506
            @test all(data.y .> 0)  # Prices should be positive
        end

        # Test Diabetes dataset
        @testset "Diabetes" begin
            data = load_diabetes()
            @test data.task == :regression
            @test size(data.X, 1) == 442
            @test size(data.X, 2) == 10
            @test length(data.y) == 442
        end

        # Test Synthetic Regression dataset
        @testset "Synthetic Regression" begin
            data = load_synthetic_reg()
            @test data.task == :regression
            @test size(data.X, 1) == 1000
            @test length(data.y) == 1000
        end
    end

    @testset "Time Series Datasets" begin
        # Test Airline Passengers dataset
        @testset "Airline Passengers" begin
            data = load_airline_passengers()
            @test data.task == :timeseries
            @test length(data.y) == 144  # 12 years * 12 months
            @test all(data.y .> 0)       # Passenger counts positive
        end

        # Test Stock Prices dataset
        @testset "Stock Prices" begin
            data = load_stock_prices()
            @test data.task == :timeseries
            @test length(data.y) == 252  # Trading days
            @test all(data.y .> 0)       # Prices positive
        end

        # Test Temperature dataset
        @testset "Temperature" begin
            data = load_temperature()
            @test data.task == :timeseries
            @test length(data.y) == 365  # Daily for 1 year
        end
    end

    @testset "List Datasets Function" begin
        datasets = list_datasets()

        # Should return a dictionary with 3 categories
        @test haskey(datasets, :classification)
        @test haskey(datasets, :regression)
        @test haskey(datasets, :timeseries)

        # Check counts
        @test length(datasets[:classification]) == 3
        @test length(datasets[:regression]) == 3
        @test length(datasets[:timeseries]) == 3

        # Check specific names
        @test "iris" in datasets[:classification]
        @test "housing" in datasets[:regression]
        @test "airline_passengers" in datasets[:timeseries]
    end

    @testset "Data Integrity" begin
        # Test that all datasets have matching X and y sizes
        all_loaders = [
            load_iris, load_titanic, load_wine_quality,
            load_housing, load_diabetes, load_synthetic_reg,
            load_airline_passengers, load_stock_prices, load_temperature
        ]

        for loader in all_loaders
            data = loader()

            # For tabular data
            if data.task in [:classification, :regression]
                @test size(data.X, 1) == length(data.y)
                @test size(data.X, 1) > 0
                @test size(data.X, 2) > 0
            end

            # For time series
            if data.task == :timeseries
                @test length(data.y) > 0
            end

            # Check no missing values in targets
            @test !any(ismissing, data.y)
        end
    end

    @testset "DataFrame Structure" begin
        # Test that loaded data has proper DataFrame structure
        data = load_iris()

        @test data.X isa DataFrame
        @test all(col -> eltype(data.X[!, col]) <: Real, names(data.X))

        # Check target
        @test data.y isa Vector
        @test length(data.y) > 0
    end

    @testset "Feature Names" begin
        # Test Iris has proper feature names
        data = load_iris()
        @test "sepal_length" in names(data.X)
        @test "sepal_width" in names(data.X)
        @test "petal_length" in names(data.X)
        @test "petal_width" in names(data.X)

        # Test Housing has proper feature names
        data_house = load_housing()
        @test "CRIM" in names(data_house.X)
        @test "RM" in names(data_house.X)
        @test "LSTAT" in names(data_house.X)
    end
end
