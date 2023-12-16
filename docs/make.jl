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
