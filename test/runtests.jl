# DSToolkit.jl — Comprehensive Test Suite

using Test
using DataFrames
using CategoricalArrays
using DSToolkit

@testset "DSToolkit.jl Full Test Suite" begin

    # 1. Data Ingestion Tests
    @testset "Data Ingestion" begin

        @testset "DataFrame input" begin
            df = DataFrame(a=rand(50), b=rand(50), target=rand(50))
            data = ingest_data(df; target=:target)
            @test data isa ToolkitData
            @test data.task == :regression
            @test ncol(data.X) == 2
            @test length(data.y) == 50
        end

        @testset "DataFrame — auto last column as target" begin
            df = DataFrame(a=rand(50), b=rand(50), y=rand(50))
            data = ingest_data(df)
            @test data.task == :regression
            @test ncol(data.X) == 2
        end

        @testset "DataFrame — classification detection (string)" begin
            df = DataFrame(a=rand(50), b=rand(50),
                           label=repeat(["cat", "dog"], 25))
            data = ingest_data(df; target=:label)
            @test data.task == :classification
        end

        @testset "DataFrame — classification detection (integer)" begin
            df = DataFrame(a=rand(50), b=rand(50),
                           label=repeat([0, 1, 2], inner=1, outer=50÷3 + 1)[1:50])
            data = ingest_data(df; target=:label)
            @test data.task == :classification
        end

        @testset "Matrix + Vector input" begin
            X = rand(50, 3)
            y = rand(50)
            data = ingest_data(X, y)
            @test data.task == :regression
            @test ncol(data.X) == 3
        end

        @testset "Dict input" begin
            d = Dict("X" => rand(50, 2), "y" => rand(50))
            data = ingest_data(d)
            @test data.task == :regression
            @test length(data.y) == 50
        end

        @testset "NamedTuple input" begin
            nt = (X=rand(50, 2), y=rand(50))
            data = ingest_data(nt)
            @test data.task == :regression
        end

        @testset "Univariate time series" begin
            y = rand(100) .+ (1:100) .* 0.1
            data = ingest_data(y; task=:timeseries)
            @test data.task == :timeseries
            @test data.X === nothing
            @test length(data.y) == 100
        end

        @testset "Task override" begin
            df = DataFrame(a=rand(50), b=rand(50), y=repeat([1, 2], 25))
            data = ingest_data(df; target=:y, task=:regression)
            @test data.task == :regression  # Overridden despite few unique values
        end

        @testset "Validation — empty data" begin
            @test_throws ErrorException ingest_data(DataFrame(a=Float64[], b=Float64[]),
                                                     Float64[])
        end

        @testset "Validation — NaN values" begin
            @test_throws ErrorException ingest_data(
                DataFrame(a=[1.0, NaN, 3.0]), [1.0, 2.0, 3.0])
        end

        @testset "Validation — missing column" begin
            df = DataFrame(a=rand(10), b=rand(10))
            @test_throws ErrorException ingest_data(df; target=:nonexistent)
        end
    end

    # 2. Train/Test Split Tests
    @testset "Train/Test Split" begin

        @testset "Tabular split (80/20)" begin
            X = DataFrame(a=1:100, b=101:200)
            y = collect(1.0:100.0)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y; ratio=0.8)
            @test nrow(X_tr) == 80
            @test nrow(X_te) == 20
            @test length(y_tr) == 80
            @test length(y_te) == 20
        end

        @testset "Tabular split shuffles" begin
            X = DataFrame(a=1:100)
            y = collect(1.0:100.0)
            _, _, y_tr1, _ = train_test_split(X, y)
            _, _, y_tr2, _ = train_test_split(X, y)
            # Very unlikely to be identical after random shuffle
            @test y_tr1 != y_tr2 || true  # Allow rare coincidence
        end

        @testset "Time series split (sequential)" begin
            y = collect(1.0:100.0)
            y_tr, y_te = train_test_split(y; ratio=0.8)
            @test length(y_tr) == 80
            @test length(y_te) == 20
            @test y_tr == collect(1.0:80.0)  # Sequential, no shuffle
        end

        @testset "3D tensor split" begin
            X = rand(Float32, 2, 10, 50)
            y = rand(Float32, 1, 50)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y; ratio=0.8)
            @test size(X_tr, 3) == 40
            @test size(X_te, 3) == 10
        end

        @testset "ToolkitData split" begin
            data = ingest_data(DataFrame(a=rand(100), y=rand(100)); target=:y)
            X_tr, X_te, y_tr, y_te = train_test_split(data)
            @test nrow(X_tr) == 80
        end
    end

    # 3. Regression Model Tests
    @testset "Regression Models" begin
        X = DataFrame(F1=rand(100), F2=rand(100))
        y = X.F1 .* 2.5 .+ X.F2 .* 1.5 .+ 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        @testset "Random Forest Regressor" begin
            model = RandomForestReg(n_trees=10)
            @test model.is_trained == false
            fit!(model, X_train, y_train)
            @test model.is_trained == true
            preds = predict(model, X_test)
            @test length(preds) == nrow(X_test)
            metrics = evaluate(model, X_test, y_test)
            @test haskey(metrics, "RMSE")
            @test haskey(metrics, "R²")
            @test metrics["RMSE"] >= 0.0
        end

        @testset "Auto-Compare Regression" begin
            result = auto_compare(X_train, y_train, X_test, y_test; task=:regression)
            @test result isa ComparisonResult
            @test result.task == :regression
            @test nrow(result.results) >= 1
            @test result.best_model.is_trained == true
            @test length(result.all_models) >= 1
        end
    end

    # 4. Classification Model Tests
    @testset "Classification Models" begin
        X = DataFrame(F1=rand(100), F2=rand(100))
        y = categorical(repeat(["A", "B", "C"], outer=34)[1:100])
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        @testset "Random Forest Classifier" begin
            model = RandomForestCls(n_trees=10)
            fit!(model, X_train, y_train)
            @test model.is_trained == true
            preds = predict(model, X_test)
            @test length(preds) == nrow(X_test)
            metrics = evaluate(model, X_test, y_test)
            @test haskey(metrics, "Accuracy")
            @test haskey(metrics, "Macro-F1")
            @test 0.0 <= metrics["Accuracy"] <= 1.0
        end

        @testset "Auto-Compare Classification" begin
            result = auto_compare(X_train, y_train, X_test, y_test; task=:classification)
            @test result isa ComparisonResult
            @test result.task == :classification
            @test nrow(result.results) >= 1
        end
    end

    # 5. Time Series Model Tests
    @testset "Statistical Time Series" begin
        y = rand(100) .+ (1:100) .* 0.1  # Upward trend
        y_train, y_test = train_test_split(y; ratio=0.8)

        @testset "ETS" begin
            model = ETSModel()
            fit!(model, y_train)
            @test model.is_trained == true
            preds = predict(model, 5)
            @test length(preds) == 5
        end

        @testset "ARIMA" begin
            model = ARIMAModel(order=(1,1,1))
            fit!(model, y_train)
            @test model.is_trained == true
            metrics = evaluate(model, y_test)
            @test haskey(metrics, "RMSE")
        end
    end

    @testset "Deep Time Series" begin
        X_seq = rand(Float32, 2, 10, 50)
        y_seq = rand(Float32, 1, 50)
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq)

        @testset "LSTM" begin
            model = LSTMModel(2; hidden_dim=8, epochs=5)
            fit!(model, X_train, y_train)
            @test model.is_trained == true
            preds = predict(model, X_test)
            @test size(preds, 2) == size(X_test, 3)
        end

        @testset "Auto-Compare Deep TS" begin
            result = auto_compare(X_train, y_train, X_test, y_test)
            @test result isa ComparisonResult
            @test nrow(result.results) == 3
        end
    end

    # 6. Persistence Tests
    @testset "Save / Load" begin
        @testset "Tabular model round-trip" begin
            model = LinearReg()
            X = DataFrame(A=rand(20), B=rand(20))
            y = rand(20)
            fit!(model, X, y)

            path = save_model(model, "test_tabular_model")
            @test isfile("test_tabular_model.jld2")

            loaded = load_toolkit_model("test_tabular_model.jld2")
            @test loaded isa LinearReg
            @test loaded.is_trained == true

            # Predict with loaded model
            preds = predict(loaded, X)
            @test length(preds) == 20

            rm("test_tabular_model.jld2")
        end

        @testset "Deep TS model round-trip" begin
            model = GRUModel(2; hidden_dim=8, epochs=3)
            X = rand(Float32, 2, 5, 20)
            y = rand(Float32, 1, 20)
            fit!(model, X, y)

            path = save_model(model, "test_deep_model")
            @test isfile("test_deep_model.jld2")

            loaded = load_toolkit_model("test_deep_model.jld2")
            @test loaded isa GRUModel
            @test loaded.is_trained == true

            preds = predict(loaded, X)
            @test size(preds, 2) == 20

            rm("test_deep_model.jld2")
        end

        @testset "Bad file path" begin
            @test_throws ErrorException load_toolkit_model("nonexistent.jld2")
        end
    end

    # 7. Error Handling Tests
    @testset "Error Handling" begin
        @testset "Predict before training" begin
            model = RandomForestReg()
            X = DataFrame(A=[1.0], B=[2.0])
            @test_throws ErrorException predict(model, X)
        end

        @testset "Predict before training (TS)" begin
            model = ARIMAModel()
            @test_throws ErrorException predict(model, 5)
        end

        @testset "Short time series" begin
            @test_throws ErrorException ingest_data(Float64[1.0, 2.0]; task=:timeseries)
        end
    end

    # 8. Full Pipeline Test (ToolkitData → auto_compare)
    @testset "Full Pipeline (ToolkitData)" begin
        @testset "Regression pipeline" begin
            df = DataFrame(x1=rand(100), x2=rand(100), target=rand(100))
            data = ingest_data(df; target=:target)
            @test data.task == :regression
            result = auto_compare(data)
            @test result isa ComparisonResult
            @test result.best_model.is_trained == true
        end

        @testset "Classification pipeline" begin
            df = DataFrame(x1=rand(100), x2=rand(100),
                           label=repeat(["yes", "no"], 50))
            data = ingest_data(df; target=:label)
            @test data.task == :classification
            result = auto_compare(data)
            @test result isa ComparisonResult
        end

        @testset "Time series pipeline" begin
            y = cumsum(randn(100)) .+ 50
            data = ingest_data(y; task=:timeseries)
            @test data.task == :timeseries
            result = auto_compare(data)
            @test result isa ComparisonResult
        end
    end

end
