using Test
using DSToolkit
using DataFrames

@testset "Inference Tests" begin

    # Setup: Train a simple model for testing
    data = load_iris()
    X_train, y_train, X_test, y_test = train_test_split(data; ratio=0.7)
    model = LogisticCls()
    fit!(model, X_train, y_train)

    @testset "Basic Inference" begin
        # Test inference function (alias for predict)
        predictions = inference(model, X_test)

        @test length(predictions) == size(X_test, 1)
        @test all(x -> x in unique(data.y), predictions)

        # Should match predict output
        predictions_predict = predict(model, X_test)
        @test predictions == predictions_predict
    end

    @testset "Inference with Preprocessing" begin
        # Create raw data and preprocessing function
        raw_data = load_housing()
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = train_test_split(
            raw_data; ratio=0.8        )

        # Define preprocessing pipeline
        function preprocess_pipeline(data)
            data = impute_missing(data; strategy=:mean)
            data = standardize(data)
            return data
        end

        # Train model on preprocessed data
        data_preprocessed = preprocess_pipeline(
            ToolkitData(X_train_raw, y_train_raw, raw_data.task)
        )
        reg_model = LinearReg()
        fit!(reg_model, data_preprocessed.X, data_preprocessed.y)

        # Test inference with preprocessing
        raw_test_data = ToolkitData(X_test_raw, y_test_raw, raw_data.task)
        predictions_with_prep = inference_with_preprocessing(
            reg_model,
            raw_test_data,
            preprocess_pipeline
        )

        @test length(predictions_with_prep) == size(X_test_raw, 1)
        @test all(isfinite, predictions_with_prep)

        # Should match manual preprocessing + predict
        preprocessed_test = preprocess_pipeline(raw_test_data)
        manual_predictions = predict(reg_model, preprocessed_test.X)
        @test predictions_with_prep ≈ manual_predictions
    end

    @testset "Batch Prediction" begin
        # Create batches
        batch_size = 10
        n_samples = size(X_test, 1)
        n_batches = div(n_samples, batch_size)

        batches = [
            X_test[(i-1)*batch_size+1:min(i*batch_size, n_samples), :]
            for i in 1:n_batches
        ]

        # Test batch prediction
        batch_results = batch_predict(model, batches)

        @test length(batch_results) == n_batches
        @test all(length(batch) > 0 for batch in batch_results)
        @test sum(length(batch) for batch in batch_results) <= n_samples

        # Concatenate and compare with single prediction
        all_batch_predictions = vcat(batch_results...)
        single_predictions = predict(model, X_test[1:length(all_batch_predictions), :])
        @test all_batch_predictions == single_predictions
    end

    @testset "Parallel Prediction" begin
        # Create chunks
        chunk_size = 15
        n_samples = size(X_test, 1)
        n_chunks = div(n_samples, chunk_size) + (n_samples % chunk_size > 0 ? 1 : 0)

        chunks = [
            X_test[(i-1)*chunk_size+1:min(i*chunk_size, n_samples), :]
            for i in 1:n_chunks
        ]

        # Test parallel prediction
        parallel_results = parallel_predict(model, chunks)

        @test length(parallel_results) == n_chunks
        @test all(length(chunk) > 0 for chunk in parallel_results)

        # Should match sequential prediction
        all_parallel_predictions = vcat(parallel_results...)
        sequential_predictions = predict(model, X_test)
        @test all_parallel_predictions == sequential_predictions
    end

    @testset "Inference on Different Data Types" begin
        # Test regression model
        reg_data = load_housing()
        X_train_r, y_train_r, X_test_r, y_test_r = train_test_split(
            reg_data; ratio=0.8
        )
        reg_model = LinearReg()
        fit!(reg_model, X_train_r, y_train_r)

        reg_predictions = inference(reg_model, X_test_r)
        @test length(reg_predictions) == size(X_test_r, 1)
        @test all(isfinite, reg_predictions)

        # Test time series model
        ts_data = load_airline_passengers()
        split_idx = Int(floor(0.8 * length(ts_data.y)))
        y_train_ts = ts_data.y[1:split_idx]

        ts_model = ARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fit!(ts_model, nothing, y_train_ts)

        ts_predictions = inference(ts_model, 12)  # 12-step ahead forecast
        @test length(ts_predictions) == 12
        @test all(isfinite, ts_predictions)
    end

    @testset "Edge Cases" begin
        # Single sample inference
        single_sample = X_test[1:1, :]
        single_prediction = inference(model, single_sample)
        @test length(single_prediction) == 1

        # Empty batch handling
        empty_batches = DataFrame[]
        try
            batch_predict(model, empty_batches)
            @test true  # Should handle gracefully
        catch e
            @test e isa ErrorException || e isa BoundsError
        end

        # Preprocessing that doesn't change data
        identity_preprocess(x) = x
        predictions_identity = inference_with_preprocessing(
            model,
            ToolkitData(X_test, y_test, data.task),
            identity_preprocess
        )
        predictions_direct = inference(model, X_test)
        @test predictions_identity == predictions_direct
    end

    @testset "Type Stability" begin
        # Classification should return vector of same type as training labels
        @test eltype(inference(model, X_test)) == eltype(y_train)

        # Regression should return Float64 vector
        reg_data = load_housing()
        X_train_r, y_train_r, X_test_r, _ = train_test_split(reg_data; ratio=0.8)
        reg_model = LinearReg()
        fit!(reg_model, X_train_r, y_train_r)
        reg_preds = inference(reg_model, X_test_r)
        @test eltype(reg_preds) <: AbstractFloat
    end

    @testset "Inference Consistency" begin
        # Multiple calls should return same results
        pred1 = inference(model, X_test)
        pred2 = inference(model, X_test)
        @test pred1 == pred2

        # Batch vs single should match
        all_at_once = inference(model, X_test)
        row_by_row = [inference(model, X_test[i:i, :])[1] for i in 1:size(X_test, 1)]
        @test all_at_once == row_by_row
    end
end
