using Documenter, SMPL

makedocs(
    sitename = "SMPL.jl",
    authors  = "Nitin",
    modules  = [SMPL],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://nitin-ppnp.github.io/SMPL.jl",
    ),
    pages = [
        "Home"               => "index.md",
        "Installation"       => "installation.md",
        "Tutorial"           => "guide.md",
        "Visualization"      => "visualization.md",
        "GPU Support"        => "gpu.md",
        "Static Compilation" => "static.md",
        "Benchmarking"       => "benchmarking.md",
        "API Reference"      => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo   = "github.com/nitin-ppnp/SMPL.jl.git",
    target = "build",
    branch = "gh-pages",
    devbranch = "master",
)
