# Generate Sample Datasets for DSToolkit
# This script generates 9 sample datasets for testing and demo purposes

using DataFrames
using CSV
using Random
using Statistics

Random.seed!(42)

println("Generating sample datasets...")

# =============================================================================
# TABULAR CLASSIFICATION DATASETS
# =============================================================================

# 1. Iris Dataset (3-class classification)
println("  [1/9] Generating iris.csv...")
n_iris = 150
iris = DataFrame(
    sepal_length = vcat(
        rand(4.5:0.1:5.8, 50),  # setosa
        rand(5.5:0.1:7.0, 50),  # versicolor
        rand(6.0:0.1:8.0, 50)   # virginica
    ),
    sepal_width = vcat(
        rand(2.8:0.1:4.5, 50),
        rand(2.0:0.1:3.4, 50),
        rand(2.5:0.1:3.8, 50)
    ),
    petal_length = vcat(
        rand(1.0:0.1:2.0, 50),
        rand(3.0:0.1:5.1, 50),
        rand(4.5:0.1:7.0, 50)
    ),
    petal_width = vcat(
        rand(0.1:0.1:0.6, 50),
        rand(1.0:0.1:1.8, 50),
        rand(1.4:0.1:2.5, 50)
    ),
    species = vcat(
        fill("setosa", 50),
        fill("versicolor", 50),
        fill("virginica", 50)
    )
)
CSV.write("data/tabular/classification/iris.csv", iris)

# 2. Titanic Dataset (binary classification)
println("  [2/9] Generating titanic.csv...")
n_titanic = 891
titanic = DataFrame(
    pclass = rand([1, 2, 3], n_titanic),
    sex = rand(["male", "female"], n_titanic),
    age = rand(1:80, n_titanic),
    sibsp = rand(0:5, n_titanic),
    parch = rand(0:4, n_titanic),
    fare = abs.(randn(n_titanic) .* 30 .+ 30),
    embarked = rand(["C", "Q", "S"], n_titanic)
)
# Survival based on realistic patterns
titanic.survived = map(eachrow(titanic)) do row
    prob = 0.4
    prob += (row.sex == "female") * 0.35
    prob += (row.pclass == 1) * 0.2
    prob += (row.age < 18) * 0.15
    prob -= (row.age > 60) * 0.15
    return rand() < prob ? 1 : 0
end
CSV.write("data/tabular/classification/titanic.csv", titanic)

# 3. Wine Quality Dataset (multi-class)
println("  [3/9] Generating wine_quality.csv...")
n_wine = 1599
wine = DataFrame(
    fixed_acidity = abs.(randn(n_wine) .* 1.5 .+ 7.0),
    volatile_acidity = abs.(randn(n_wine) .* 0.15 .+ 0.4),
    citric_acid = abs.(randn(n_wine) .* 0.15 .+ 0.25),
    residual_sugar = abs.(randn(n_wine) .* 2.0 .+ 2.5),
    chlorides = abs.(randn(n_wine) .* 0.03 .+ 0.08),
    free_sulfur_dioxide = abs.(randn(n_wine) .* 10.0 .+ 15.0),
    total_sulfur_dioxide = abs.(randn(n_wine) .* 30.0 .+ 40.0),
    density = randn(n_wine) .* 0.002 .+ 0.996,
    pH = randn(n_wine) .* 0.15 .+ 3.3,
    sulphates = abs.(randn(n_wine) .* 0.15 .+ 0.65),
    alcohol = abs.(randn(n_wine) .* 1.0 .+ 10.5)
)
# Quality score 3-8 based on features
wine.quality = map(eachrow(wine)) do row
    score = 5.0
    score += (row.alcohol - 10.5) * 0.3
    score += (row.volatile_acidity - 0.4) * (-2.0)
    score += (row.citric_acid - 0.25) * 1.5
    score = clamp(round(Int, score), 3, 8)
    return score
end
CSV.write("data/tabular/classification/wine_quality.csv", wine)

# =============================================================================
# TABULAR REGRESSION DATASETS
# =============================================================================

# 4. Housing Dataset (Boston housing prices)
println("  [4/9] Generating housing.csv...")
n_housing = 506
housing = DataFrame(
    CRIM = abs.(randn(n_housing) .* 5.0 .+ 3.0),      # crime rate
    ZN = rand(0.0:0.1:100.0, n_housing),               # residential land zoned
    INDUS = rand(0.0:0.1:30.0, n_housing),             # non-retail business acres
    CHAS = rand([0, 1], n_housing),                    # bounds Charles River
    NOX = rand(0.3:0.01:0.9, n_housing),               # nitric oxide concentration
    RM = randn(n_housing) .* 0.8 .+ 6.3,               # avg rooms per dwelling
    AGE = rand(0.0:100.0, n_housing),                  # prop built pre-1940
    DIS = abs.(randn(n_housing) .* 2.0 .+ 4.0),       # distances to employment centers
    RAD = rand(1:24, n_housing),                       # accessibility to highways
    TAX = rand(200:700, n_housing),                    # property tax rate
    PTRATIO = rand(12.0:0.1:22.0, n_housing),          # pupil-teacher ratio
    B = rand(300.0:400.0, n_housing),                  # proportion of Black residents
    LSTAT = abs.(randn(n_housing) .* 5.0 .+ 12.0)     # % lower status population
)
# Price based on realistic patterns
housing.PRICE = map(eachrow(housing)) do row
    price = 22.0
    price += (row.RM - 6.3) * 8.0
    price -= (row.LSTAT - 12.0) * 0.5
    price -= (row.CRIM - 3.0) * 0.3
    price += (row.DIS - 4.0) * 0.8
    price -= (row.PTRATIO - 17.0) * 1.2
    price += row.CHAS * 5.0
    price = max(price + randn() * 3.0, 5.0)
    return round(price, digits=1)
end
CSV.write("data/tabular/regression/housing.csv", housing)

# 5. Diabetes Dataset (medical regression)
println("  [5/9] Generating diabetes.csv...")
n_diabetes = 442
diabetes = DataFrame(
    age = rand(19:80, n_diabetes),
    sex = rand([0, 1], n_diabetes),
    bmi = randn(n_diabetes) .* 5.0 .+ 26.0,
    bp = randn(n_diabetes) .* 10.0 .+ 120.0,
    s1 = abs.(randn(n_diabetes) .* 20.0 .+ 190.0),   # total cholesterol
    s2 = abs.(randn(n_diabetes) .* 15.0 .+ 115.0),   # LDL
    s3 = abs.(randn(n_diabetes) .* 10.0 .+ 50.0),    # HDL
    s4 = abs.(randn(n_diabetes) .* 1.0 .+ 4.5),      # total/HDL cholesterol
    s5 = randn(n_diabetes) .* 0.5 .+ 4.9,            # log triglycerides
    s6 = abs.(randn(n_diabetes) .* 15.0 .+ 90.0)     # blood sugar
)
# Disease progression
diabetes.target = map(eachrow(diabetes)) do row
    prog = 150.0
    prog += (row.bmi - 26.0) * 4.0
    prog += (row.bp - 120.0) * 0.8
    prog += (row.s1 - 190.0) * 0.3
    prog += (row.s6 - 90.0) * 0.5
    prog += (row.age - 50) * 0.5
    prog = max(prog + randn() * 20.0, 25.0)
    return round(Int, prog)
end
CSV.write("data/tabular/regression/diabetes.csv", diabetes)

# 6. Synthetic Regression Dataset (clean linear data)
println("  [6/9] Generating synthetic_reg.csv...")
n_synth = 1000
synthetic = DataFrame(
    feature_1 = randn(n_synth) .* 10.0,
    feature_2 = randn(n_synth) .* 5.0,
    feature_3 = randn(n_synth) .* 8.0,
    feature_4 = randn(n_synth) .* 3.0,
    feature_5 = randn(n_synth) .* 6.0
)
# Clean linear relationship: y = 2*x1 + 1.5*x2 - 0.8*x3 + 3*x4 + 0.5*x5 + noise
synthetic.target = (
    2.0 .* synthetic.feature_1 .+
    1.5 .* synthetic.feature_2 .-
    0.8 .* synthetic.feature_3 .+
    3.0 .* synthetic.feature_4 .+
    0.5 .* synthetic.feature_5 .+
    randn(n_synth) .* 2.0
)
CSV.write("data/tabular/regression/synthetic_reg.csv", synthetic)

# =============================================================================
# TIME SERIES DATASETS
# =============================================================================

# 7. Airline Passengers (classic seasonal TS)
println("  [7/9] Generating airline_passengers.csv...")
n_months = 144
months = 1:n_months
trend = 100.0 .+ (months .- 1) .* 1.2
seasonality = 20.0 .* sin.(2π .* months ./ 12)
noise = randn(n_months) .* 5.0
airline = DataFrame(
    month = months,
    passengers = round.(Int, trend .+ seasonality .+ noise)
)
CSV.write("data/timeseries/airline_passengers.csv", airline)

# 8. Stock Prices (financial TS with random walk + drift)
println("  [8/9] Generating stock_prices.csv...")
n_days = 252
initial_price = 100.0
drift = 0.0005  # small upward drift
volatility = 0.02
returns = drift .+ volatility .* randn(n_days)
prices = initial_price .* cumprod(1.0 .+ returns)
stock = DataFrame(
    day = 1:n_days,
    price = round.(prices, digits=2)
)
CSV.write("data/timeseries/stock_prices.csv", stock)

# 9. Temperature (weather TS with annual cycle)
println("  [9/9] Generating temperature.csv...")
n_days_temp = 365
days_temp = 1:n_days_temp
base_temp = 15.0  # base temperature in Celsius
annual_cycle = 10.0 .* sin.(2π .* days_temp ./ 365 .- π/2)  # peak in summer
noise_temp = randn(n_days_temp) .* 2.5
temperature = DataFrame(
    day = days_temp,
    temperature = round.(base_temp .+ annual_cycle .+ noise_temp, digits=1)
)
CSV.write("data/timeseries/temperature.csv", temperature)

println("\n✅ All 9 datasets generated successfully!")
println("\nDataset Summary:")
println("  Classification:")
println("    - iris.csv (150 rows, 4 features, 3 classes)")
println("    - titanic.csv (891 rows, 7 features, binary)")
println("    - wine_quality.csv (1599 rows, 11 features, 6 classes)")
println("  Regression:")
println("    - housing.csv (506 rows, 13 features)")
println("    - diabetes.csv (442 rows, 10 features)")
println("    - synthetic_reg.csv (1000 rows, 5 features)")
println("  Time Series:")
println("    - airline_passengers.csv (144 months)")
println("    - stock_prices.csv (252 days)")
println("    - temperature.csv (365 days)")
