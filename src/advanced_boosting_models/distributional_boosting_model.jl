export DistributionalBoostingModel

import Distributions.logpdf

"""
```
DistributionalBoostingModel{T<:AbstractFloat, D<:Distributions.Distribution}(
 	dist::Type{D},
 	boosters::Vector{RootBoostingModel{T}},
 	transform::ParameterizableTransform
)
```

`dist` can be any `Distributions.Distribution`, the output of `transform` ``\\circ`` `boosters` then needs to provide valid parameters to that distribution.

Example:
```
using AdvancedBoosting

import Distributions.Normal

model = DistributionalBoostingModel(
    Normal,
    [RootBoostingModel(2,5),RootBoostingModel(2,5)],
    MultiTransform([IdentityTransform([1]), SoftplusTransform([2])])
)
```
The respective probabilistic model is then

```math
p(y|\\mathbf{x})=\\mathcal{N}(y|f_1(\\mathbf{x}),s(f_2(\\mathbf{x}))
```

where ``f_1,f_2`` are individual Gradient Boosting models and ``s`` is the softplus transform:

```math
\\text{softplus}(x)=\\log \\left(\\exp(x)+1\\right)
```
"""
mutable struct DistributionalBoostingModel{T<:AbstractFloat,D<:Distribution} <:
               AdvancedBoostingModel{T}
    dist::Type{D}
    boosters::Vector{RootBoostingModel{T}}
    transform::ParameterizableTransform
end

function (m::DistributionalBoostingModel{T,D})(
    X::Matrix{T},
) where {T<:AbstractFloat,D<:Distribution}
    predictions = m.boosters(X)
    preds_us = unstack(predictions, 1)
    mapped_predictions = map(pred -> m.dist(m.transform(pred)...), preds_us)

    return vcat(mapped_predictions)
end

function model_loss(
    m::DistributionalBoostingModel{T,D},
    X::Vector{T},
    transform::ParameterizableTransform,
    ypred,
    y::Union{T,AbstractArray{T}},
) where {T<:AbstractFloat,D<:Distribution}
    return -logpdf(m.dist(transform(ypred)...), y)
end
