module AdvancedBoosting

include("transforms.jl")
export IdentityTransform,
	SoftplusTransform,
	SigmoidTransform,
	MultiTransform,
	VaryingCoefficientTransform,
        ComposedTransform

include("root_boosting_model.jl")
export RootBoostingModel,
       	is_trained

include("advanced_boosting_models/advanced_boosting_model.jl")
export AdvancedBoostingModel,
       fit!

include("advanced_boosting_models/distributional_boosting_model.jl")
export DistributionalBoostingModel

include("advanced_boosting_models/varying_coefficient_boosting_model.jl")
export VaryingCoefficientBoostingModel

end
