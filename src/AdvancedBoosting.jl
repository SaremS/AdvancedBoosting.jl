module AdvancedBoosting

include("transforms.jl")
export IdentityTransform,
	SoftplusTransform,
	SigmoidTransform,
	MultiTransform

include("root_boosting_model.jl")
export RootBoostingModel,
       	is_trained

end
