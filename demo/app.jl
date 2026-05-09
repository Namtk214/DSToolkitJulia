# DSToolkit Demo Web UI
# Built with Genie.jl

# To run this demo:
# 1. Install Genie: using Pkg; Pkg.add("Genie")
# 2. Run: julia --project=. demo/app.jl
# 3. Open browser: http://localhost:8000

using Genie
using DSToolkit

# Configuration
const PORT = 8000

# Initialize Genie app
Genie.config.run_as_server = true
Genie.config.server_port = PORT

# Routes
route("/") do
    "DSToolkit Demo - Coming Soon! Install Genie.jl to enable full UI."
end

route("/api/datasets") do
    # List available datasets
    datasets = [
        "iris", "titanic", "wine_quality",
        "housing", "diabetes", "synthetic_reg",
        "airline_passengers", "stock_prices", "temperature"
    ]
    return JSON.json(Dict("datasets" => datasets))
end

println("DSToolkit Demo Server")
println("=====================")
println("Server will start on http://localhost:$PORT")
println("\nNote: Full UI requires Genie.jl to be installed:")
println("  using Pkg; Pkg.add(\"Genie\")")

# Start server (requires Genie to be installed)
try
    up(PORT, async=false)
catch e
    println("\nError: Genie.jl not installed. Install it with:")
    println("  using Pkg; Pkg.add(\"Genie\")")
end
