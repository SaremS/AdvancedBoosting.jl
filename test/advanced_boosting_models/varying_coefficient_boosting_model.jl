using Test, AdvancedBoosting

using DecisionTree, Random


@testset "Simple Varying Coefficient model" begin
    model = VaryingCoefficientBoostingModel(
        [
            RootBoostingModel(1.0, Float64[], DecisionTreeRegressor[], 1, 0),
            RootBoostingModel(2.0, Float64[], DecisionTreeRegressor[], 1, 0),
        ],
        VaryingCoefficientTransform(),
    )

    prediction = model(ones(1, 2))

    @test prediction == [4.0]
end

@testset "VaryingCoefficientBoosting init_intercepts! runs through" begin
    model = VaryingCoefficientBoostingModel(
        [RootBoostingModel(nothing, Float64[], DecisionTreeRegressor[], 1, 0)],
        VaryingCoefficientTransform(),
    )

    X = collect(-3:0.1:3)[:, :]

    Random.seed!(123)
    y = sin.(X)[:] .+ randn(size(X, 1))

    AdvancedBoosting.init_intercepts!(model, X, y)
end

@testset "DistributionalBoosting build_trees! runs through" begin
    model = VaryingCoefficientBoostingModel(
        [RootBoostingModel(1.0, Float64[], DecisionTreeRegressor[], 1, 0)],
        VaryingCoefficientTransform(),
    )

    X = collect(-3:0.1:3)[:, :]

    Random.seed!(123)
    y = sin.(X)[:] .+ randn(size(X, 1))

    AdvancedBoosting.build_trees!(model, X, y)
end
