using Test, AdvancedBoosting

using Distributions, DecisionTree

@testset "Simple root booster" begin
    model =
        RootBoostingModel(0.0, Float64[], DecisionTreeRegressor[], 1, 0, target_dims = [1])

    result = model(zeros(2))
    @test result == [0.0]

    result = model(zeros(2, 3))
    @test result == [0.0, 0.0]
end

@testset "Simple root booster with tree" begin
    tree = DecisionTreeRegressor(max_depth = 1)
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    y = zeros(3)
    DecisionTree.fit!(tree, X, y)

    model = RootBoostingModel(
        1.0,
        Float64[1.0],
        DecisionTreeRegressor[tree],
        1,
        1,
        target_dims = [1, 2],
    )

    result = model(zeros(2))
    @test result == [1.0] .+ predict(tree, zeros(1, 2))

    result = model(zeros(2, 2))
    @test result == [1.0, 1.0] .+ predict(tree, zeros(2, 2))
end

@testset "Two simple root booster" begin
    models = [
        RootBoostingModel(0.0, Float64[], DecisionTreeRegressor[], 1, 0, target_dims = [1]),
        RootBoostingModel(1.0, Float64[], DecisionTreeRegressor[], 1, 0, target_dims = [1]),
    ]

    result = models(zeros(2))
    @test result == [0.0 1.0]

    result = models(zeros(2, 3))
    @test result == [0.0 1.0; 0.0 1.0]
end
