using Documenter, AdvancedBoosting

doctest(AdvancedBoosting)
makedocs(
    sitename = "AdvancedBoosting.jl",
    modules = [AdvancedBoosting],
    pages = [
        "index.md",
        "Advanced Boosting Models" => [
            "AdvancedBoostingModel" => "advanced_boosting_models/advanced_boosting_model.md",
	    "Distributional Boosting" => "advanced_boosting_models/distributional_boosting_model.md"
        ],
        "Parameterizable Transformations" => "parameterizable_transform.md",
        "Root Boosting Model" => "root_boosting_model.md",
    ],
)

# Generate Gaussian example
using Pkg
Pkg.add(["Plots", "Distributions"])
using AdvancedBoosting, Random, Plots

import Distributions.Normal, Distributions.mean, Distributions.std

Random.seed!(321);

X = rand(100,1) .* 6 .- 3;
y = sin.(X) .+ randn(100,1) .* (0.25 .* abs.(sin.(X)) .+ 0.1);


model = DistributionalBoostingModel(
    Normal,
    [RootBoostingModel(1,3),RootBoostingModel(1,3)],
    MultiTransform([IdentityTransform([1]), SoftplusTransform([2])])
);

fit!(model, X, y[:])

lines = collect(-3:0.1:3)[:,:];
pred_dists = model(lines);

p1 = plot();

plot!(p1,lines[:], sin.(lines[:]), ribbon = 2 .* (0.25 .* sin.(lines[:]).^2 .+ 0.1), label="Ground truth distribution", title="Gaussian Distribution with sin(x) mean + variance", 
    titlefontsize=10, fmt=:png, lw=2);
plot!(p1, lines[:],mean.(pred_dists), ribbon = 2 .* std.(pred_dists), label="Gradient Boosting estimate", lw=2);
scatter!(p1, X[:],y[:], markersize = 0.25, label = "Data (n=100)");

savefig(p1, "docs/build/assets/normdist_example.png");
