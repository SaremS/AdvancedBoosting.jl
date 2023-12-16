export RootBoostingModel, is_trained

import DecisionTree.DecisionTreeRegressor

"""
Single boosting model whose inputs, outputs and parameters are of type `T<:Real`.

Can be constructed either via

```julia
RootBoostingModel{T}(max_depth::Int64, n_trees::Int64)::RootBoostingModel{T} where T<:Real
```

or

```julia
RootBoostingModel(max_depth::Int64, n_trees::Int64)::RootBoostingModel{Float64}
```

where `max_depth` defines the maximum depth of each tree (typically, `max_depth=1` suffices in a Gradient Boosting model) and `n_trees` defines the number of trees to include in the model, i.e. the number refinement iterations

Once a `is_trained(RootBoostingModel{T})==true`, i.e. the model has been fitted to some data, we can give it an input matrix `X::Matrix{T}` to produce some predictions:

```julia
model = RootBoostingModel(1,100)
...(training the model here)
X = randn(5,10) #5 datapoints with 10 features each
predictions = model(X)
```

It is also possible to use an input vector `X::Vector{T}`. In that case, `X` will be treated as a single-row matrix, i.e. an input matrix with a single data point.

```julia
model = RootBoostingModel(1,100)
...(training the model here)
X = randn(10) # a single datapoint with 10 features
predictions = model(X)
```

Notice that the `RootBoostingModel` deliberately has no API to be trained directly. Rather, a higher level model typically employs several `RootBoostingModel`s and the training happens via the higher level model APIs.
"""
mutable struct RootBoostingModel{T<:Real}
    intercept::Union{T,Nothing}
    coeffs::Vector{T}
    trees::Vector{DecisionTreeRegressor}

    max_depth::Int64
    n_trees::Int64
end


function RootBoostingModel{T}(max_depth::Int64, n_trees::Int64) where {T<:Real}
    return RootBoostingModel(
        nothing,
        T[],
        DecisionTreeRegressor[],
        transform,
        max_depth,
        n_trees,
    )
end

function RootBoostingModel(max_depth::Int64, n_trees::Int64)
    return RootBoostingModel{Float64}(
        nothing,
        Float64[],
        DecisionTreeRegressor[],
        max_depth,
        n_trees,
    )
end

"""
```julia
is_trained(m::RootBoostingModel)::Bool
```

Check whether a boosting model has already been fitted/trained.
"""
function is_trained(m::RootBoostingModel)::Bool
    !isnothing(m.intercept)
end

function (m::RootBoostingModel{T})(X::Matrix{T})::Vector{T} where {T<:Real}
    @assert is_trained(m)

    n, _ = size(X)
    n_trees = length(m.trees)

    predictions = zeros(n)

    predictions .+= m.intercept .* ones(n)

    if n_trees > 0
        tree_predictions = map(i -> m.coeffs[i] .* predict(m.trees[i], X), 1:n_trees)
        predictions .+= sum(tree_predictions)
    end

    return predictions
end

function (m::RootBoostingModel{T})(X::Vector{T})::Vector{T} where {T<:Real}
    Xmat = Matrix(transpose(X))
    return m(Xmat)
end