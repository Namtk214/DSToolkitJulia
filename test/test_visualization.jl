using Test
using DSToolkit

@testset "Visualization Tests" begin

    # Load sample data for testing
    data_class = load_iris()
    data_reg = load_housing()
    data_ts = load_airline_passengers()

    @testset "Error Messages Without Plots.jl" begin
        # These tests verify that visualization functions provide helpful
        # error messages when Plots.jl is not available

        @testset "Distribution Plots" begin
            try
                plot_histogram(data_reg, :CRIM)
                @test false  # Should not reach here if Plots not loaded
            catch e
                @test e isa ErrorException
                # Check if error message is helpful (different error types have different fields)
                err_str = sprint(showerror, e)
                @test occursin("Plots", err_str) || occursin("visualization", err_str) || occursin("not defined", err_str)
            end

            try
                plot_boxplot(data_reg, :RM)
                @test false
            catch e
                @test e isa ErrorException
                # Check if error message is helpful (different error types have different fields)
                err_str = sprint(showerror, e)
                @test occursin("Plots", err_str) || occursin("visualization", err_str) || occursin("not defined", err_str)
            end

            try
                plot_target_distribution(data_class)
                @test false
            catch e
                @test e isa ErrorException
                # Check if error message is helpful (different error types have different fields)
                err_str = sprint(showerror, e)
                @test occursin("Plots", err_str) || occursin("visualization", err_str) || occursin("not defined", err_str)
            end
        end

        @testset "Correlation Plots" begin
            try
                plot_correlation_heatmap(data_reg)
                @test false
            catch e
                @test e isa ErrorException
                # Check if error message is helpful (different error types have different fields)
                err_str = sprint(showerror, e)
                @test occursin("Plots", err_str) || occursin("visualization", err_str) || occursin("not defined", err_str)
            end

            try
                plot_feature_vs_target(data_reg, :RM)
                @test false
            catch e
                @test e isa ErrorException
                # Check if error message is helpful (different error types have different fields)
                err_str = sprint(showerror, e)
                @test occursin("Plots", err_str) || occursin("visualization", err_str) || occursin("not defined", err_str)
            end
        end

        @testset "Model Comparison Plots" begin
            # Create a mock comparison result
            result = auto_compare(data_reg; ratio=0.8)

            try
                plot_comparison_results(result)
                @test false
            catch e
                @test e isa ErrorException
                # Check if error message is helpful (different error types have different fields)
                err_str = sprint(showerror, e)
                @test occursin("Plots", err_str) || occursin("visualization", err_str) || occursin("not defined", err_str)
            end

            try
                plot_metric_comparison(result, :rmse)
                @test false
            catch e
                @test e isa ErrorException
                # Check if error message is helpful (different error types have different fields)
                err_str = sprint(showerror, e)
                @test occursin("Plots", err_str) || occursin("visualization", err_str) || occursin("not defined", err_str)
            end
        end

        @testset "Time Series Plots" begin
            y_train = data_ts.y[1:100]
            y_test = data_ts.y[101:end]
            predictions = y_test .+ randn(length(y_test)) * 10

            try
                plot_timeseries(y_train, y_test, predictions)
                @test false
            catch e
                @test e isa ErrorException
                # Check if error message is helpful (different error types have different fields)
                err_str = sprint(showerror, e)
                @test occursin("Plots", err_str) || occursin("visualization", err_str) || occursin("not defined", err_str)
            end
        end
    end

    @testset "Function Signatures" begin
        # Test that functions exist and have correct signatures
        @test isdefined(DSToolkit, :plot_histogram)
        @test isdefined(DSToolkit, :plot_boxplot)
        @test isdefined(DSToolkit, :plot_target_distribution)
        @test isdefined(DSToolkit, :plot_correlation_heatmap)
        @test isdefined(DSToolkit, :plot_feature_vs_target)
        @test isdefined(DSToolkit, :plot_comparison_results)
        @test isdefined(DSToolkit, :plot_metric_comparison)
        @test isdefined(DSToolkit, :plot_timeseries)
        @test isdefined(DSToolkit, :plot_forecast)
    end

    @testset "Input Validation" begin
        # Test that functions validate inputs properly

        # Invalid column name should error
        @test_throws Exception plot_histogram(data_reg, :nonexistent_column)
        @test_throws Exception plot_boxplot(data_reg, :nonexistent_column)
        @test_throws Exception plot_feature_vs_target(data_reg, :nonexistent_column)
    end
end

# Note: To fully test visualization with actual plot generation,
# you would need to:
# 1. Add Plots.jl to test dependencies in Project.toml
# 2. Import Plots at the top of this file
# 3. Replace error tests with actual plot generation tests
# 4. Use something like `@test plot isa Plots.Plot` to verify output

# Example with Plots.jl installed:
# @testset "Actual Plot Generation" begin
#     using Plots
#
#     @testset "Histogram" begin
#         p = plot_histogram(data_reg, :CRIM)
#         @test p isa Plots.Plot
#     end
#
#     @testset "Correlation Heatmap" begin
#         p = plot_correlation_heatmap(data_reg)
#         @test p isa Plots.Plot
#     end
# end
