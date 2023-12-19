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


#Generate censoring example
import Distributions.censored
Random.seed!(321)

X = rand(200,1) .* 6 .- 3
f(x) = censored(Normal(sin( 2 * x), 0.25*abs(sin(x))+0.1), lower=0.0)
y = rand.(f.(X))

#define custom distribution to match our type definition
import Distributions.ContinuousUnivariateDistribution, Distributions.logpdf, Distributions.mean, Distributions.quantile


struct ZeroCensoredNormal <: ContinuousUnivariateDistribution
    mu
    sigma
end

logpdf(m::ZeroCensoredNormal, y) = logpdf(censored(Normal(m.mu, m.sigma), lower=0.0), y)
mean(m::ZeroCensoredNormal) = mean(censored(Normal(m.mu, m.sigma), lower=0.0))
quantile(m::ZeroCensoredNormal, p) = quantile(censored(Normal(m.mu, m.sigma), lower=0.0),p)


model = DistributionalBoostingModel(
    ZeroCensoredNormal,
    [RootBoostingModel(1,5),RootBoostingModel(1,5)],
    MultiTransform([IdentityTransform([1]), SoftplusTransform([2])])
)

fit!(model, X, y[:])

lines = collect(-3:0.1:3)[:,:]
pred_dists = model(lines)
mean_pred = mean.(pred_dists)
ribbon_pred = (mean_pred .- quantile.(pred_dists,0.05), quantile.(pred_dists,0.95) .- mean_pred)

line_dists = f.(lines)
mean_line = mean.(line_dists)
ribbon_line = (mean_line .- quantile.(line_dists,0.05), quantile.(line_dists,0.95) .- mean_line)


p1 = plot()

plot!(p1,lines[:], mean_line, ribbon = ribbon_line, label="Ground truth distribution", title="Gaussian Distribution with harmonic mean + variance, censored at 0.0", 
    titlefontsize=10, fmt=:png, lw=2)
plot!(p1, lines[:], mean_pred, ribbon = ribbon_pred, label="Gradient Boosting estimate", lw=2)
scatter!(p1, X[:],y[:], markersize = 0.25, label = "Data (n=200)")

savefig(p1, "docs/build/assets/normdist_censored_example.png");
