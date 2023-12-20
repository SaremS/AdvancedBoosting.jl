using AdvancedBoosting
using Test

const tests = [
    "advanced_boosting_models/distributional_boosting_model",
    "transforms",
    "root_boosting_model",
]

@testset "AdvancedBoosting.jl" begin
    @testset "Test $t" for t in tests
        include("$t.jl")
    end
end
