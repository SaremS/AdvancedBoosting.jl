export AdvancedBoostingModel,
       fit!,
       init_coeffs!,
       build_trees!

"""
Abstract type that specifies the behavior of all 'advanced boosting models'.

These models are meant to consist of multiple standard boosting models (`RootBoostingModel`) and combine them into a single, non-trivial and combined model.

Each model needs to be associated with a respective prediction function 
`(m::AdvancedBoostingModel{T})(X::Matrix{T}) where {T<:Real}` and the `init_coeffs!` and `build_trees!` functions.

The former initializes the ``\\alpha_0`` intercepts of each root model, whereas the latter fits the decision trees of the subsequent models. The `fit!` function then provides a single interface for any `AdvancedBoostinModel` by subsequently calling both `init_coeffs!` and `build_trees!`.
"""
abstract type AdvancedBoostingModel{T<:Real} end

(m::AdvancedBoostingModel{T})(X::Matrix{T}) where {T<:Real} = @error "Not implemented"

function fit!(m::AdvancedBoostingModel{T}, X::Matrix{T}, y::Vector{T}) where {T<:Real}
    init_coeffs!(m, X, y)
    build_trees!(m, X, y)
end

init_coeffs!(m::AdvancedBoostingModel{T}, X::Matrix{T}, y::Vector{T}) where {T<:Real} =
    @error "Not implemented"
build_trees!(m::AdvancedBoostingModel{T}, X::Matrix{T}, y::Vector{T}) where {T<:Real} =
    @error "Not implemented"
