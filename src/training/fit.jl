# Training Module - fit! implementations
# This file imports all model-specific fit! methods

# Import utilities
include("utils.jl")

# Regression models
include("../models/regression/linear.jl")
include("../models/regression/tree_based.jl")
include("../models/regression/distance_based.jl")

# Classification models
include("../models/classification/logistic.jl")
include("../models/classification/tree_based.jl")
include("../models/classification/distance_based.jl")

# Time series models
include("../models/timeseries/statistical.jl")
include("../models/timeseries/deep_learning.jl")
