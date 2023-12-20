export VaryingCoefficientBoostingModel

import Flux.unstack


"""
```
VaryingCoefficientBoostingModel{T<:AbstractFloat}(
	boosters::Vector{RootBoostingModel{T}},
	transform::VaryingCoefficientTransform
)
```

Example:

```
using AdvancedBoosting

using Random

model = VaryingCoefficientModel(
	[RootBoostingModel(1,5), RootBoostingModel(1,5)],
	VaryingCoefficientTransform
)

Random.seed!(123)
X = randn(100,2)
y = sin.(X[:,1]) .+ cos(X[:,2])

fit!(model, X, y)
```

Currently, this model is trained via the MSE-criterion. Other criteria can easily be added
"""
mutable struct VaryingCoefficientBoostingModel{T<:AbstractFloat} <: AdvancedBoostingModel{T}
    boosters::Vector{RootBoostingModel{T}}
    transform::VaryingCoefficientTransform
end

function (m::VaryingCoefficientBoostingModel{T})(X::Matrix{T}) where {T<:AbstractFloat}
    predictions = m.boosters(X)
    preds_us = unstack(predictions, 1)
    mapped_predictions = map(i->m.transform(preds_us[i],X[i,:]), 1:length(preds_us))
    
    return vcat(mapped_predictions)
end

function model_loss(
	m::VaryingCoefficientBoostingModel{T},
        X::Vector{T},
        transform::ParameterizableTransform,
        ypred,
        y::Union{T, AbstractArray{T}}) where {T<:AbstractFloat}
     return (transform(ypred, X)[1] - y[1])^2
end
