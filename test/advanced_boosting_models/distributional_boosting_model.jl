using Test, AdvancedBoosting

using Distributions, DecisionTree


@testset "DistributionalBoosting loss" begin
    model = DistributionalBoostingModel(
        Normal,
        [
            RootBoostingModel(0.0, Float64[], DecisionTreeRegressor[], 1, 0),
            RootBoostingModel(1.0, Float64[], DecisionTreeRegressor[], 1, 0),
        ],
        MultiTransform([IdentityTransform([1]), IdentityTransform([2])]),
    )

    #loss = AdvancedBoosting.model_loss(model, zeros(5), model.transform, 1.0, 0.0)

    #@test loss == logpdf(Normal(0.0, 1.0), 0.0)
end
