export ParameterizableTransform,
    IdentityTransform,
    SoftplusTransform,
    SigmoidTransform,
    MultiTransform,
    VaryingCoefficientTransform

import ReverseDiff.TrackedReal
import Flux.@functor

"""
The abstract supertype corresponding to parameterizable transformations.
"""
abstract type ParameterizableTransform end

#Use `AbstractVector` here instead of `Vector{T} where {T<:Real}` since autodiff also requires
#to pass through `TrackedArray`s which would clash with the latter.
(t::ParameterizableTransform)(boosting_output) = @error "Not implemented"
(t::ParameterizableTransform)(boosting_output, X) = @error "Not implemented"

"""
Applies the identity transformation,

```math
g(f(\\mathbf{x}), \\mathbf{x})=f(\\mathbf{x})_{[i_1,i_2,...,i_m]},
```

where ``[i_1,i_2,...,i_m]`` denotes the indices defined in `target_idx`


```jldoctest
using AdvancedBoosting
transform = IdentityTransform([1])
transform([2.], zeros(3))

# output

1-element Vector{Float64}:
 2.0
```

```jldoctest
using AdvancedBoosting
transform = IdentityTransform([1])
transform([2.])

# output

1-element Vector{Float64}:
 2.0
```
"""
mutable struct IdentityTransform <: ParameterizableTransform
    target_idx::Vector{Int64}
end

function (t::IdentityTransform)(boosting_output)
    return boosting_output[t.target_idx]
end

function (t::IdentityTransform)(boosting_output, X::Union{TrackedReal,AbstractVector})
    return boosting_output[t.target_idx]
end


"""
Applies the softplus transform,

```math
g(f(\\mathbf{x}),\\mathbf{x})=\\log(\\exp(f(\\mathbf{x})_{[i_1,i_2,...,i_m]})+1),
```

where ``[i_1,i_2,...,i_m]`` denotes the elements defined in `target_idx`.

```jldoctest
using AdvancedBoosting
transform = SoftplusTransform([1])
transform([0.], zeros(3))

# output

1-element Vector{Float64}:
 0.6931471805599453
```

```jldoctest
using AdvancedBoosting
transform = SoftplusTransform([1])
transform([0.])

# output

1-element Vector{Float64}:
 0.6931471805599453
```
"""
mutable struct SoftplusTransform <: ParameterizableTransform
    target_dims::Vector{Int64}
end

function (t::SoftplusTransform)(boosting_output)
    return softplus.(boosting_output[t.target_dims])
end

function (t::SoftplusTransform)(boosting_output, X)
    return softplus.(boosting_output[t.target_dims])
end

softplus(x::T) where {T<:Real} = log(exp(x) + 1.0)


"""
Applies the sigmoid transform,

```math
g(f(\\mathbf{x}),\\mathbf{x})=\\frac{1}{1+\\exp(-f(\\mathbf{x})_{[i_1,i_2,...,i_m]})},
```

where ``[i_1,i_2,...,i_m]`` denotes the elements defined in `target_idx`.

```jldoctest
using AdvancedBoosting
transform = SigmoidTransform([1])
transform([0.], zeros(3))

# output

1-element Vector{Float64}:
 0.5
```

```jldoctest
using AdvancedBoosting
transform = SigmoidTransform([1])
transform([0.])

# output

1-element Vector{Float64}:
 0.5
```
"""
mutable struct SigmoidTransform <: ParameterizableTransform
    target_dims::Vector{Int64}
end

function (t::SigmoidTransform)(boosting_output)
    return sigmoid.(boosting_output[t.target_dims])
end

function (t::SigmoidTransform)(boosting_output, X)
    return sigmoid.(boosting_output[t.target_dims])
end

sigmoid(x::T) where {T<:Real} = 1.0 / (1.0 + exp(-x))


"""
Applies multiple `ParameterizableTransform`s on the same input and concatenates their outputs. I.e. let ``g_1(\\cdot,\\cdot),...,g_n(\\cdot,\\cdot)`` denote a set of ``n`` `ParameterizableTransform`s. Then, the ``MultiTransform`` computes as

```math
g(f(\\mathbf{x}),\\mathbf{x})=\\left(g_1(f(\\mathbf{x}),\\mathbf{x}),...,g_n(f(\\mathbf{x}),\\mathbf{x})\\right)^T,
```

where the outputs of each individual transform are flattened.

```jldoctest
using AdvancedBoosting
transform1 = IdentityTransform([1])
transform2 = SigmoidTransform([2])
transform = MultiTransform([transform1, transform2])

transform([0.,0.], zeros(3))

# output

2-element Vector{Float64}:
 0.0
 0.5
```

```jldoctest
using AdvancedBoosting
transform1 = IdentityTransform([1])
transform2 = SigmoidTransform([2])
transform = MultiTransform([transform1, transform2])

transform([0.,0.])

# output

2-element Vector{Float64}:
 0.0
 0.5
```
"""
mutable struct MultiTransform <: ParameterizableTransform
    transforms::Vector{ParameterizableTransform}
end

function (t::MultiTransform)(boosting_output, X)
    return vcat(map(transform -> transform(boosting_output, X), t.transforms)...)
end

function (t::MultiTransform)(boosting_output)
    return vcat(map(transform -> transform(boosting_output), t.transforms)...)
end


"""
Applies the outputs of multiple gradient boosting models as a varying coefficient model.
I.e., for an input vector ``\\mathbf{x}\\in\\mathbb{R}^M`` and ``M`` stacked boosting models,

```math
f(\\cdot)=\\left(f_1(\\cdot),...,f_M(\\cdot)\\right)^T,
```
we have

```math
g(f(\\mathbf{x}),\\mathbf{x};\\alpha)=\\alpha + \\left(f_1(\\cdot),...,f_M(\\cdot)\\right) \\mathbf{x},
```

where ``\\alpha`` denotes the intercept of the varying coefficient model.

```jldoctest
using AdvancedBoosting
transform = VaryingCoefficientTransform()

transform([1.,2.], [1.,2.])

# output

6.0
```
"""
mutable struct VaryingCoefficientTransform <: ParameterizableTransform
    intercept::AbstractVector
end
@functor VaryingCoefficientTransform

VaryingCoefficientTransform() = VaryingCoefficientTransform(ones(1))

function (t::VaryingCoefficientTransform)(
    boosting_output::AbstractVector,
    X::AbstractVector,
)
    return t.intercept[1] .+ transpose(boosting_output) * X
end
