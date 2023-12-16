export AdvancedBoostingModel

import Distributions.Distribution

abstract type AdvancedBoostingModel end

mutable struct DistributionalBoostingModel{T<:Real, D<:Distribution} <: AbstractBoostingModel 
    dist::Type{D}
    boosters::Vector{RootBoostingModel{T}}
    transform::ParameterizableTransform
end

function (models::Vector{RootBoostingModel{T}})(X::Matrix{T}) where T<:Real
    return hcat(map(booster->booster(X), models)...)
end


function (m::DistributionalBoostingModel{T})(X::Matrix{T}) where T<:Real
    predictions = m.boosters(X)
    preds_us = Flux.unstack(predictions,1)
    mapped_predictions = map(pred->m.dist(m.transform(pred)...), preds_us)
    
    return vcat(mapped_predictions)
end
