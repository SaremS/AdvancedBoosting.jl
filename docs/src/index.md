# AdvancedBoosting.jl

Experimental package for various Gradient Boosting models.

## Some examples

### Gaussian conditional distribution
Here, we model the conditional mean and standard deivation as Gradient Boosting models, i.e.

```math
p(y|\mathbf{x})=\mathcal{N}(y|f_1(\mathbf{x}),s(f_2(\mathbf{x})))
```

where $f_1,f_2$ are individual Gradient Boosting models and $s$ is the softplus function

```math
\text{softplus}(x)=\log\left(\exp(x)\right)
```

```julia
using AdvancedBoosting, Random, Plots
import Distributions.Normal, Distributions.mean, Distributions.std

Random.seed!(321);

X = rand(100,1) .* 6 .- 3;
y = sin.(X) .+ randn(100,1) .* (0.25 .* abs.(sin.(X)) .+ 0.1);


model = DistributionalBoostingModel(
    Normal, #conditional distribution shoud be normal
    [RootBoostingModel(1,3), RootBoostingModel(1,3)], #both conditional mean and standard deviation are modelled by GradientBoosting
    MultiTransform(
        [IdentityTransform([1]), #mean model output stay as is
         SoftplusTransform([2])  #stddev model output is mapped to the positive, non-zero reals
    ]) 
);

fit!(model, X, y[:])

lines = collect(-3:0.1:3)[:,:];
pred_dists = model(lines);

p1 = plot();

plot!(p1
    lines[:], sin.(lines[:]),
    ribbon = 2 .* (0.25 .* sin.(lines[:]).^2 .+ 0.1),
    label="Ground truth distribution",
    title="Gaussian Distribution with sin(x) mean + variance", 
    titlefontsize=10,
    fmt=:png,
    lw=2
);
plot!(p1, 
    lines[:],mean.(pred_dists),
    ribbon = 2 .* std.(pred_dists),
    label="Gradient Boosting estimate", lw=2
);
scatter!(p1,
    X[:],y[:],
    markersize = 0.25,
    label = "Data (n=100)"
);
```

![](assets/normdist_example.png)
