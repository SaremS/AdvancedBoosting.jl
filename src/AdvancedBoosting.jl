module AdvancedBoosting

include("transforms.jl")
export IdentityTransform,
	SoftplusTransform,
	SigmoidTransform,
	MultiTransform

include("root_boosting_model.jl")
export RootBoostingModel,
       	is_trained

include("advanced_boosting_models/advanced_boosting_model.jl")
export AdvancedBoostingModel,
       fit!

include("advanced_boosting_models/distributional_boosting_model.jl")
export DistributionalBoostingModel

end
