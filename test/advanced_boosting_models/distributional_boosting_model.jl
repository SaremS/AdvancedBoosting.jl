using Test, AdvancedBoosting

using Distributions, DecisionTree, Random


@testset "DistributionalBoosting loss" begin
    model = DistributionalBoostingModel(
        Normal,
        [
            RootBoostingModel(
                0.0,
                Float64[],
                DecisionTreeRegressor[],
                1,
                0,
                target_dims = [1],
            ),
            RootBoostingModel(
                1.0,
                Float64[],
                DecisionTreeRegressor[],
                1,
                0,
                target_dims = [1],
            ),
        ],
        MultiTransform([IdentityTransform([1]), IdentityTransform([2])]),
    )

    loss = AdvancedBoosting.model_loss(model, zeros(5), model.transform, [0.0, 1.0], 0.0)
    @test loss == -logpdf(Normal(0.0, 1.0), 0.0)

    stacked_loss = AdvancedBoosting.model_loss_stacked(
        model,
        zeros(2, 3),
        model.transform,
        [0.0 1.0; 1.0 2.0],
        [0.0, 1.0],
    )
    @test stacked_loss == -logpdf.([Normal(0.0, 1.0), Normal(1.0, 2.0)], [0.0, 1.0])
end

@testset "DistributionalBoosting init_intercepts! runs through" begin
    model = DistributionalBoostingModel(
        Normal,
        [
            RootBoostingModel(nothing, Float64[], DecisionTreeRegressor[], 1, 0),
            RootBoostingModel(nothing, Float64[], DecisionTreeRegressor[], 1, 0),
        ],
        MultiTransform([IdentityTransform([1]), SoftplusTransform([2])]),
    )

    X = collect(-3:0.1:3)[:, :]

    Random.seed!(123)
    y = sin.(X)[:] .+ randn(size(X, 1))

    AdvancedBoosting.init_intercepts!(model, X, y)
end

@testset "DistributionalBoosting build_trees! runs through" begin
    model = DistributionalBoostingModel(
        Normal,
        [
            RootBoostingModel(
                0.0,
                Float64[],
                DecisionTreeRegressor[],
                1,
                1,
                target_dims = [1],
            ),
            RootBoostingModel(
                0.0,
                Float64[],
                DecisionTreeRegressor[],
                1,
                1,
                target_dims = [1],
            ),
        ],
        MultiTransform([IdentityTransform([1]), SoftplusTransform([2])]),
    )

    X = collect(-3:0.1:3)[:, :]

    Random.seed!(123)
    y = sin.(X)[:] .+ randn(size(X, 1))

    AdvancedBoosting.build_trees!(model, X, y)
end

@testset "DistributionalBoosting fit! runs through" begin
    model = DistributionalBoostingModel(
        Normal,
        [
            RootBoostingModel(nothing, Float64[], DecisionTreeRegressor[], 1, 0),
            RootBoostingModel(nothing, Float64[], DecisionTreeRegressor[], 1, 0),
        ],
        MultiTransform([IdentityTransform([1]), SoftplusTransform([2])]),
    )

    X = collect(-3:0.1:3)[:, :]

    Random.seed!(123)
    y = sin.(X)[:] .+ randn(size(X, 1))

    AdvancedBoosting.fit!(model, X, y)
end
