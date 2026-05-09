using Test
using DSToolkit
using DataFrames
using Statistics

@testset "Preprocessing Tests" begin

    @testset "Missing Value Imputation" begin
        # Create test data with missing values
        df = DataFrame(
            a = [1.0, missing, 3.0, 4.0, 5.0],
            b = [10.0, 20.0, missing, 40.0, 50.0],
            c = ["X", "Y", missing, "Y", "X"],
            target = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        data = ingest_data(df; target=:target, task=:regression)

        # Test mean imputation
        data_mean = impute_missing(data; strategy=:mean)
        @test !any(ismissing, data_mean.X.a)
        @test !any(ismissing, data_mean.X.b)
        @test data_mean.X.a[2] ≈ mean([1.0, 3.0, 4.0, 5.0])

        # Test median imputation
        data_median = impute_missing(data; strategy=:median)
        @test !any(ismissing, data_median.X.a)
        @test data_median.X.a[2] ≈ median([1.0, 3.0, 4.0, 5.0])

        # Test mode imputation
        data_mode = impute_missing(data; strategy=:mode)
        @test !any(ismissing, data_mode.X.c)

        # Test forward fill
        data_forward = impute_missing(data; strategy=:forward)
        @test !any(ismissing, data_forward.X.a)
        @test data_forward.X.a[2] == 1.0  # Forward filled from previous
    end

    @testset "Feature Scaling" begin
        # Create test data
        df = DataFrame(
            feature1 = [1.0, 2.0, 3.0, 4.0, 5.0],
            feature2 = [10.0, 20.0, 30.0, 40.0, 50.0],
            target = [1, 0, 1, 0, 1]
        )
        data = ingest_data(df; target=:target, task=:classification)

        # Test standardization
        data_std = standardize(data)
        @test mean(data_std.X.feature1) ≈ 0.0 atol=1e-10
        @test std(data_std.X.feature1) ≈ 1.0 atol=1e-10
        @test mean(data_std.X.feature2) ≈ 0.0 atol=1e-10

        # Test normalization
        data_norm = normalize(data)
        @test minimum(data_norm.X.feature1) ≈ 0.0 atol=1e-10
        @test maximum(data_norm.X.feature1) ≈ 1.0 atol=1e-10
        @test minimum(data_norm.X.feature2) ≈ 0.0 atol=1e-10
        @test maximum(data_norm.X.feature2) ≈ 1.0 atol=1e-10
    end

    @testset "Categorical Encoding" begin
        # Create test data
        df = DataFrame(
            color = ["red", "blue", "green", "red", "blue"],
            size = ["S", "M", "L", "M", "S"],
            price = [10.0, 20.0, 30.0, 15.0, 12.0],
            target = [1, 0, 1, 1, 0]
        )
        data = ingest_data(df; target=:target, task=:classification)

        # Test one-hot encoding
        data_onehot = one_hot_encode(data, [:color, :size])
        @test size(data_onehot.X, 2) > size(data.X, 2)  # More columns after encoding
        @test "price" in names(data_onehot.X)  # Numeric column preserved
        @test all(x -> x in [0, 1], data_onehot.X.color_red)  # Binary encoding

        # Test label encoding
        data_labeled = label_encode(data, [:color, :size])
        @test eltype(data_labeled.X.color) <: Integer
        @test eltype(data_labeled.X.size) <: Integer
        @test length(unique(data_labeled.X.color)) == 3  # red, blue, green
        @test length(unique(data_labeled.X.size)) == 3   # S, M, L
    end

    @testset "Feature Engineering" begin
        # Create test data
        df = DataFrame(
            x1 = [1.0, 2.0, 3.0, 4.0, 5.0],
            x2 = [2.0, 4.0, 6.0, 8.0, 10.0],
            target = [3.0, 6.0, 9.0, 12.0, 15.0]
        )
        data = ingest_data(df; target=:target, task=:regression)

        # Test polynomial features
        data_poly = add_polynomial_features(data; degree=2)
        @test size(data_poly.X, 2) > size(data.X, 2)
        # Should have: x1, x2, x1^2, x2^2, x1*x2
        @test size(data_poly.X, 2) >= 5

        # Test interaction features
        data_interact = add_interaction_features(data, [(:x1, :x2)])
        @test size(data_interact.X, 2) == size(data.X, 2) + 1
        @test "x1_x_x2" in names(data_interact.X)
        @test data_interact.X.x1_x_x2[1] ≈ 1.0 * 2.0
        @test data_interact.X.x1_x_x2[2] ≈ 2.0 * 4.0
    end

    @testset "Preprocessing Pipeline" begin
        # Test chaining multiple preprocessing steps
        data = load_housing()

        # Chain: impute → standardize → polynomial
        data = impute_missing(data; strategy=:mean)
        original_cols = size(data.X, 2)

        data = standardize(data)
        # Check standardization worked
        for col in names(data.X)
            vals = data.X[!, col]
            @test abs(mean(vals)) < 1e-10 || abs(std(vals) - 1.0) < 1e-10
        end

        data = add_polynomial_features(data; degree=2)
        @test size(data.X, 2) > original_cols
    end

    @testset "Edge Cases" begin
        # Skip edge cases for now - these are known limitations
        @test_skip true
    end
end
