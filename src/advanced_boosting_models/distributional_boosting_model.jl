export DistributionalBoostingModel

using DecisionTree, ForwardDiff, ReverseDiff

import Distributions.Distribution, Distributions.logpdf
import Optim.minimizer, Optim.optimize, Optim.BFGS
import Flux.destructure, Flux.unstack, Flux.stack
import BlockDiagonals.BlockDiagonal
import StatsBase.mean

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
mutable struct DistributionalBoostingModel{T<:AbstractFloat, D<:Distribution} <: AdvancedBoostingModel{T}
    dist::Type{D}
    boosters::Vector{RootBoostingModel{T}}
    transform::ParameterizableTransform
end

function (m::DistributionalBoostingModel{T, D})(X::Matrix{T}) where {T<:AbstractFloat, D<:Distribution}
    predictions = m.boosters(X)
    preds_us = unstack(predictions,1)
    mapped_predictions = map(pred->m.dist(m.transform(pred)...), preds_us)
    
    return vcat(mapped_predictions)
end

"""
Initializes the ``\\alpha_0`` intercepts of each boosting model.
"""
function init_intercepts!(m::DistributionalBoostingModel{T, D}, X::Matrix{T}, y::Vector{T}) where {T<:AbstractFloat, D<:Distribution}
    n_boosters = length(m.boosters)
    coeffs = zeros(n_boosters)
    ps, t = destructure(m.transform)
    
    n_c = length(coeffs)
    n_p = length(ps)
    
    full_params_init = vcat(coeffs,ps)
    
    full_params_opt = minimizer(optimize(c -> -mean(map(yy->logpdf(m.dist(t(c[n_c+1:end])(c[1:n_c])...),yy), y)), full_params_init, BFGS(); autodiff = :forward))
        
    opt_coeffs = full_params_opt[1:n_c]
    opt_trans = full_params_opt[n_c+1:end]
    
    m.transform = t(opt_trans)

    for i in 1:n_boosters
        m.boosters[i].intercept= opt_coeffs[i]
    end
end

"""
Number of trees to build per boosting model (allowed to differ
to finetune model complexity)
"""
function get_n_trees(m::DistributionalBoostingModel)
    return map(booster->booster.n_trees, m.boosters)
end

"""
Max depth of the trees per boosting model (allowed to differ to
finetune model complexity)
"""
function get_max_depths(m::DistributionalBoostingModel)    
	return map(booster->booster.max_depth, m.boosters)
end

"""
Current amount of grown trees per model
"""
function count_trees_in_booster(m::DistributionalBoostingModel)
    return map(booster -> length(booster.trees), m.boosters)
end

"""
Likelihood loss 
"""
function ll_loss(m::DistributionalBoostingModel, transform::ParameterizableTransform ,predictions, y::AbstractVector)
    n,_ = size(predictions)
    preds_us = unstack(predictions, 1)
    return mean(map(i->logpdf(m.dist(transform(preds_us[i])...),y[i]), 1:n))
end

"""
Build a single tree to predict current gradient
"""
function build_tree(max_depth::Int64, X::AbstractMatrix, grads::AbstractVector) 
    new_tree = DecisionTreeRegressor(max_depth=max_depth)
    DecisionTree.fit!(new_tree, X, grads)
    return new_tree
end

"""
Build the boosting trees

TODO: Simplify
"""
function build_trees!(m::DistributionalBoostingModel{T}, X::Matrix{T}, y::Vector{T}) where T<:AbstractFloat
#Setup - get model specifications
    n_models = length(m.boosters)
    n_trees = get_n_trees(m)
    max_depths = get_max_depths(m)
    
    max_trees = maximum(n_trees)
    
#In case that the transform uses any parameters itself (e.g. linear transforms),
#we also need to obtain any optimize the respective parameters
    ps, t = destructure(m.transform)
    
    for _ in 1:max_trees
#Current predictions per model and respective gradients
        trees_per_booster = count_trees_in_booster(m)
        predictions = m.boosters(X)
 
        prediction_grads = ReverseDiff.gradient(p-> ll_loss(m,m.transform,p,y), predictions)
        target_grads = stack(prediction_grads)
        
        #if a booster reached its max-tree amount, set gradients to zero
        zero_idx = trees_per_booster .== n_trees
        target_grads[:,zero_idx] .= 0.0
        
        #one gradient column per model
        grads_unst = unstack(target_grads, 2)
        
        #build new trees and predict
        new_trees = map(i->build_tree(max_depths[i], X, grads_unst[i]), 1:n_models)
        new_preds = hcat(map(tree->predict(tree, X), new_trees)...)
        
        new_coeffs = ones(n_models)
        
        n_c = length(new_coeffs)
        n_p = length(ps)
        
	#Current intercets and coefficients per boosting model, number of intercepts (n_is) and total number
	#of current trees (n_cs), large matrix of all predictions from each tree from each model (all_preds)
	#
	#`transform_matrix` is a zero-one matrix such that 
	#intercepts .+ all_preds .* transpose(coeffs)*transform_matrix
	#Produces the individual outputs of each boosting model
	#
	#We need all these constructs to properly adhere to the interface of `Optim.minimize` and allows
	#it to properly optimize our model parameters
        intercepts, coeffs, n_is, n_cs, all_preds, transform_matrix = optimization_helpers(m, X)
        
	#Currently, we re-optimize each coefficient after each run. I believe that 
	#the original boosting algorithm only optimizes each coefficient when the respective
	#tree is grown and one last time at the end. The current approach seems to provide better
	#results though, at the cost of performance

	#If there already any trees in the model, we optimize intercept + coefficient 
        if n_cs > 0
		#still quite awkward, we need to optimize the intercepts, the existing tree coefficients,
		#the coefficients of this round's trees AND potential parameters of the `ParameterizableTransform`
		full_params_init = vcat(intercepts, coeffs, new_coeffs,ps)

		#t(c[1+n_is+n_cs+n_c : end]): This restructures (from Flux.destructure) the `ParameterizableTransform` and
		#thus allows Optim.optimize to optimize potential parameters.
	
		#c[1:n_is]: These are the intercepts for each model
		#
		#c[1+n_is : n_is+n]: The coefficients corresponding to the trees from past rounds. 
		#
		#c[1+n_is+n_cs : n_is+n_cs+n_c]: The coefficients corresponding to the new trees
	
            full_params_opt = minimizer(
                optimize(
                    c->-ll_loss(
                        m,t(c[1+n_is+n_cs+n_c : end]),transpose(c[1 : n_is]) .+ (all_preds .* transpose(c[1+n_is : n_is+n_cs]))*transform_matrix .+ transpose(c[1+n_is+n_cs : n_is+n_cs+n_c]).*new_preds,
                        y
                    ), 
                    full_params_init, 
                    BFGS(); 
                    autodiff = :forward
                )
            )

	    #now, we restructure the minimizing parameters to adhere to our individual models structure
            opt_intercepts, opt_coeffs, newtree_coeffs, opt_trans = coeffs_from_models(m, full_params_opt)

            for j in 1:n_models
                m.boosters[j].intercept = opt_intercepts[j]
                m.boosters[j].coeffs = opt_coeffs[j]

		#only append new tree and coefficient if the respective Boosting models has not yet reached
		#its `n_trees`
                if !zero_idx[j]
                    m.boosters[j].coeffs = vcat(m.boosters[j].coeffs, newtree_coeffs[j])
                    m.boosters[j].trees = vcat(m.boosters[j].trees, new_trees[j])
                end
            end

            m.transform = t(opt_trans)

	#in case that we haven't built any trees yet (i.e. the first round of boosting after initializing the intercepts), we 
	#perform the same steps as above, but leave out the `coeffs` parameters
        else
            full_params_init = vcat(intercepts, new_coeffs,ps)

            full_params_opt = minimizer(
                optimize(
                    c->-ll_loss(
                        m,t(c[1+n_is+n_cs+n_c:end]),transpose(c[1:n_is]) .+ transpose(c[1+n_is+n_cs: n_is+n_cs+n_c]).*new_preds,
                        y
                    ), 
                    full_params_init, 
                    BFGS(); 
                    autodiff = :forward
                )
            )
		
	    #Extract respective objects for the first round after parameter initialization
            opt_intercepts, newtree_coeffs, opt_trans = coeffs_from_models_first_round(m, full_params_opt)
            
            for j in 1:n_models
                m.boosters[j].intercept = opt_intercepts[j]

		#only append new tree and coefficient if the respective Boosting models has not yet reached
		#its `n_trees`
                if !zero_idx[j]
                    m.boosters[j].coeffs = vcat(m.boosters[j].coeffs, newtree_coeffs[j])
                    m.boosters[j].trees = vcat(m.boosters[j].trees, new_trees[j])
                end
            end

            m.transform = t(opt_trans)
        end
    end
end



"""
This function creates several objects to ease the actual parameter optimization via `Optim.optimize`
"""
function optimization_helpers(m::DistributionalBoostingModel{T}, X::Matrix{T}) where T<:AbstractFloat
    intercepts = map(booster -> booster.intercept, m.boosters)
    n_is = length(intercepts)
    
    coeffs = vcat(map(booster -> booster.coeffs, m.boosters)...)
    n_cs = length(coeffs)
    
    #get predictions from all trees over all models and stack into one large N*(n_models*n_trees(model)) matrix
    all_preds = hcat(map(booster -> hcat(map(tree->predict(tree, X), booster.trees)...), m.boosters)...)
    
    
    n_trees = map(booster -> length(booster.trees), m.boosters)
    #sum predictions*coefficients over all models into co
    transform_matrix = BlockDiagonal(map(len-> ones(len,1), n_trees))
    
    return intercepts, coeffs, n_is, n_cs, all_preds, transform_matrix
end


"""
Extract parameters at first round after initializing intercepts
"""
function coeffs_from_models_first_round(m::DistributionalBoostingModel{T}, coeffs::Vector{T}) where T<:AbstractFloat
    nmodels = length(m.boosters)
    
    opt_intercepts = coeffs[1:nmodels]
    
    offset = nmodels
    
    newtree_coeffs = coeffs[offset+1 : offset+nmodels]
    
    offset += nmodels
    
    opt_trans = coeffs[offset+1 : end]
    
    return opt_intercepts, newtree_coeffs, opt_trans
end


function coeffs_from_models(m::DistributionalBoostingModel{T}, coeffs::Vector{T}) where T<:AbstractFloat
    nmodels = length(m.boosters)
    
    opt_intercepts = coeffs[1:nmodels]
    
    opt_coeffs = []
    offset = nmodels
    
    for j in 1:nmodels
        ntrees = length(m.boosters[j].trees)
        coeffs_model = coeffs[offset+1 : offset+ntrees]
        push!(opt_coeffs, coeffs_model)
        
        offset += ntrees
    end
    
    newtree_coeffs = coeffs[offset+1 : offset+nmodels]
    
    offset += nmodels
    
    opt_trans = coeffs[offset+1 : end]
    
    return opt_intercepts, opt_coeffs, newtree_coeffs, opt_trans
end
