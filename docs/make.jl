using Documenter, AdvancedBoosting

doctest(AdvancedBoosting)
makedocs(
    sitename = "AdvancedBoosting.jl",
    modules = [AdvancedBoosting],
    pages = [
        "index.md",
        "AdvancedBoostingModels" => [
            "AdvancedBoostingModel" => "advanced_boosting_models/advanced_boosting_model.md",
	    "Distributional Boosting" => "advanced_boosting_models/distributional_boosting_model.md"
        ],
        "Parameterizable Transformations" => "parameterizable_transform.md",
        "Root Boosting Model" => "root_boosting_model.md",
    ],
)
