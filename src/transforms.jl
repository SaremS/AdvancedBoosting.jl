export ParameterizableTransform,
    IdentityTransform,
    SoftplusTransform,
    SigmoidTransform,
    MultiTransform,
    VaryingCoefficientTransform,
    ComposedTransform,
    SumTransform

import ReverseDiff.TrackedReal
import Flux.@functor
import Base.+

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
g(f(\\mathbf{x}),\\mathbf{x};\\alpha)=\\alpha + \\left(f_{(i_1)}(\\mathbf{x}),...,f__{(i_K)}(\\mathbf{x})\\right)^T \\mathbf{x}_{(i_1,...,i_K)},
```

where ``\\alpha`` denotes the intercept of the varying coefficient model and ``i_1,...,i_K`` denote the indices defined in `target_dims`

```jldoctest
using AdvancedBoosting
transform = VaryingCoefficientTransform([1,2])

transform([1.,2.], [1.,2.])

# output

1-element Vector{Float64}:
 6.0
```
"""
mutable struct VaryingCoefficientTransform <: ParameterizableTransform
    intercept::AbstractVector
    target_dims_boosters::Vector{Int64}
    target_dims_x::Vector{Int64}
end
@functor VaryingCoefficientTransform (intercept,)

VaryingCoefficientTransform(target_dims::Vector{Int64}) =
    VaryingCoefficientTransform(ones(1), target_dims, target_dims)

VaryingCoefficientTransform(target_dims_boosters::Vector{Int64}, target_dims_x::Vector{Int64}) =
    VaryingCoefficientTransform(ones(1), target_dims_boosters, target_dims_x)

function (t::VaryingCoefficientTransform)(
    boosting_output::AbstractVector,
    X::AbstractVector,
)
    return t.intercept .+ transpose(boosting_output[t.target_dims_boosters]) * X[t.target_dims_x]
end


"""
Composes two `ParameterizableTransform`s as follows:

Let ``g1,g2`` denote two `ParameterizableTransform`s, ``\\mathbf{x}`` some input vector and
``\\mathbf{f}(\\mathbf{x})`` the outputs of one or more boosting models, then

```math
g(\\mathbf{f}(\\mathbf{x}),\\mathbf{x})=(g_2\\circ g_1)(\\mathbf{f}(\\mathbf{x}),\\mathbf{x})=g_2(g_1(\\mathbf{f}(\\mathbf{x}),\\mathbf{x}),\\mathbf{x})
```

Example:

```jldoctest
using AdvancedBoosting
g1 = VaryingCoefficientTransform([1,2])
g2 = SoftplusTransform([1])

transform = g2∘g1

transform([1.,2.], [1.,2.])

# output

1-element Vector{Float64}:
 6.00247568513773
```
"""
mutable struct ComposedTransform <: ParameterizableTransform
    g1::ParameterizableTransform
    g2::ParameterizableTransform
end

function (t::ComposedTransform)(boosting_output, X)
    return t.g2(t.g1(boosting_output, X), X)
end

function ∘(g2::ParameterizableTransform, g1::ParameterizableTransform)
    return ComposedTransform(g1, g2)
end


"""
Sums two `ParameterizableTransform`s as follows:

Let ``g1,g2`` denote two `ParameterizableTransform`s, ``\\mathbf{x}`` some input vector and
``\\mathbf{f}(\\mathbf{x})`` the outputs of one or more boosting models, then

```math
g(\\mathbf{f}(\\mathbf{x}),\\mathbf{x})=(g_1 + g_2)(\\mathbf{f}(\\mathbf{x}),\\mathbf{x})=g_1(\\mathbf{f}(\\mathbf{x}),\\mathbf{x}) + g_2(\\mathbf{f}(\\mathbf{x}),\\mathbf{x})
```

Example:

```jldoctest
using AdvancedBoosting
g1 = SoftplusTransform([1])
g2 = SoftplusTransform([1])

transform = g2+g1

transform([1.], [1.])

# output

1-element Vector{Float64}:
 2.6265233750364456
```
"""
mutable struct SumTransform <: ParameterizableTransform
    g1::ParameterizableTransform
    g2::ParameterizableTransform
end

function (t::SumTransform)(boosting_output, X)
    return t.g1(boosting_output, X) .+ t.g1(boosting_output, X)
end

function +(g1::ParameterizableTransform, g2::ParameterizableTransform)
    return SumTransform(g1, g2)
end
