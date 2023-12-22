export RootBoostingModel, is_trained, set_target_dims!

import DecisionTree.DecisionTreeRegressor

"""
Single boosting model whose inputs, outputs and parameters are of type `T<:AbstractFloat`.

Can be constructed either via

```julia
RootBoostingModel{T}(max_depth::Int64, n_trees::Int64)::RootBoostingModel{T} where T<:AbstractFloat
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

As a helper function, it is also possible to make predictions from a vector of `RootBoostingModel`s as follows:

```julia
models = [RootBoostingModel(1,100), RootBoostingModel(1,100)]
...(training the models here)
X = randn(10) # a single datapoint with 10 features
predictions = models(X) #returns a vector of prediction vectors
```
"""
mutable struct RootBoostingModel{T<:AbstractFloat}
    intercept::Union{T,Nothing}
    coeffs::Vector{T}
    trees::Vector{DecisionTreeRegressor}

    max_depth::Int64
    n_trees::Int64

    target_dims::Union{Nothing,Vector{Int64}}
end

function RootBoostingModel{T}(
    intercept::Union{T,Nothing},
    coeffs::Vector{T},
    trees::Vector{DecisionTreeRegressor},
    max_depth::Int64,
    n_trees::Int64,
) where {T<:AbstractFloat}
    return RootBoostingModel{T}(intercept, coeffs, trees, max_depth, n_trees, nothing)
end

function RootBoostingModel(
    intercept::Union{Float64,Nothing},
    coeffs::Vector{Float64},
    trees::Vector{DecisionTreeRegressor},
    max_depth::Int64,
    n_trees::Int64,
)
    return RootBoostingModel{Float64}(intercept, coeffs, trees, max_depth, n_trees)
end

function RootBoostingModel(
    intercept::Union{Float64,Nothing},
    coeffs::Vector{Float64},
    trees::Vector{DecisionTreeRegressor},
    max_depth::Int64,
    n_trees::Int64;
    target_dims::Union{Nothing, Vector{Int64}} = nothing
)
    return RootBoostingModel{Float64}(intercept, coeffs, trees, max_depth, n_trees, target_dims)
end

function RootBoostingModel{T}(
    max_depth::Int64,
    n_trees::Int64;
    target_dims = nothing,
) where {T<:AbstractFloat}
    return RootBoostingModel{T}(
        nothing,
        T[],
        DecisionTreeRegressor[],
        max_depth,
        n_trees,
        target_dims,
    )
end

function RootBoostingModel(max_depth::Int64, n_trees::Int64; target_dims = nothing)
    return RootBoostingModel{Float64}(
        nothing,
        Float64[],
        DecisionTreeRegressor[],
        max_depth,
        n_trees,
        target_dims,
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


function set_target_dims!(m::RootBoostingModel, target_dims::Vector{Int64})
    if isnothing(m.target_dims)
        m.target_dims = target_dims
    end
end

function set_target_dims!(m::RootBoostingModel, n_dims::Int64)
    set_target_dims!(m, collect(1:n_dims))
end

function (m::RootBoostingModel{T})(X::Matrix{T})::Vector{T} where {T<:AbstractFloat}
    @assert is_trained(m)

    n, _ = size(X)
    n_trees = length(m.trees)

    predictions = zeros(n)

    predictions .+= m.intercept .* ones(n)

    X_target = X[:, m.target_dims]

    if n_trees > 0
        tree_predictions = map(i -> m.coeffs[i] .* predict(m.trees[i], X_target), 1:n_trees)
        predictions .+= sum(tree_predictions)
    end

    return predictions
end

function (m::RootBoostingModel{T})(X::Vector{T})::Vector{T} where {T<:AbstractFloat}
    Xmat = Matrix(transpose(X[m.target_dims]))
    return m(Xmat)
end

function (models::Vector{RootBoostingModel{T}})(X::Matrix{T}) where {T<:AbstractFloat}
    return hcat(map(booster -> booster(X), models)...)
end

function (models::Vector{RootBoostingModel{T}})(X::Vector{T}) where {T<:AbstractFloat}
    return hcat(map(booster -> booster(X), models)...)
end
