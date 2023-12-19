export DistributionalBoostingModel

import Distributions.Distribution

"""
```
DistributionalBoostingModel{T<:Real, D<:Distributions.Distribution}(
	dist::Type{D},
	boosters::Vector{RootBoostingModel{T}},
	transform::ParameterizableTransform
)
```

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
"""
mutable struct DistributionalBoostingModel{T<:Real, D<:Distribution} <: AdvancedBoostingModel{Real}
    dist::Type{D}
    boosters::Vector{RootBoostingModel{T}}
    transform::ParameterizableTransform
end
