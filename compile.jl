using JuliaC

isdir("build") && rm("build"; recursive=true)
mkpath("build")

img = ImageRecipe(
    output_type = "--output-exe",
    file        = "staticSMPL.jl",
    project     = "static_project",
    trim_mode   = "unsafe",
    add_ccallables = false,
    verbose     = true,
)

link = LinkRecipe(
    image_recipe = img,
    outname      = "build/smpl",
    rpath        = nothing, # set automatically when bundling
)

bun = BundleRecipe(
    link_recipe = link,
    output_dir  = "build", # or `nothing` to skip bundling
)

compile_products(img)
link_products(link)
bundle_products(bun)
