using Documenter, AdvancedBoosting

doctest(AdvancedBoosting)
makedocs(
    sitename = "AdvancedBoosting.jl",
    modules = [AdvancedBoosting],
    pages = ["index.md", 
	     "Parameterizable Transformations" => "parameterizable_transform.md",
	     "Root Boosting Model" => "root_boosting_model.md"
	],
)

deploydocs(
    repo = "github.com/SaremS/AdvancedBoosting.jl.git",
)
