"""
One-time conversion script: SMPL .npz → custom binary format (.smplbin)

Run this in a normal Julia environment (with NPZ available):
    julia scripts/convert_model.jl [path/to/SMPL_MALE.npz] [output/SMPL_MALE.smplbin]

Binary format (all little-endian):
  For each of 7 arrays in order:
    [v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights, faces]
    - ndims::UInt8         (number of dimensions)
    - shape::UInt64 × ndims  (each dimension size)
    - dtype::UInt8         (1=Float32, 2=UInt32)
    - data::dtype × prod(shape)  (raw row-major data)
"""

using NPZ

function write_array(io::IO, arr::Array{Float32})
    write(io, UInt8(ndims(arr)))
    for d in size(arr)
        write(io, UInt64(d))
    end
    write(io, UInt8(1))  # Float32
    write(io, arr)
end

function write_array(io::IO, arr::Array{UInt32})
    write(io, UInt8(ndims(arr)))
    for d in size(arr)
        write(io, UInt64(d))
    end
    write(io, UInt8(2))  # UInt32
    write(io, arr)
end

function convert_npz_to_bin(npz_path::String, out_path::String)
    println("Reading: $npz_path")
    model = NPZ.npzread(npz_path)

    v_template  = Float32.(model["v_template"])          # (6890, 3)
    shapedirs   = Float32.(reshape(model["shapedirs"], 6890*3, :))  # (6890*3, 300)
    posedirs    = Float32.(reshape(model["posedirs"], 6890*3, :)')  # (207, 6890*3)  transposed
    J_regressor = Float32.(model["J_regressor"])         # (24, 6890)
    parents     = UInt32.(model["kintree_table"][1, :])  # (24,)
    lbs_weights = Float32.(model["weights"])             # (6890, 24)
    faces       = UInt32.(model["f"] .+ 1)               # (faces, 3) — python→julia indexing

    println("v_template:  ", size(v_template))
    println("shapedirs:   ", size(shapedirs))
    println("posedirs:    ", size(posedirs))
    println("J_regressor: ", size(J_regressor))
    println("parents:     ", size(parents))
    println("lbs_weights: ", size(lbs_weights))
    println("faces:       ", size(faces))

    open(out_path, "w") do io
        # Magic header
        write(io, UInt8[0x53, 0x4D, 0x50, 0x4C])  # "SMPL"
        write(io, UInt32(1))                        # version

        write_array(io, v_template)
        write_array(io, shapedirs)
        write_array(io, posedirs)
        write_array(io, J_regressor)
        write_array(io, parents)
        write_array(io, lbs_weights)
        write_array(io, faces)
    end

    sz = filesize(out_path)
    println("Written: $out_path  ($(round(sz/1024/1024, digits=2)) MB)")
end

# ---- Main ----
default_in  = joinpath(homedir(), ".julia", "scratchspaces",
                       "124859b0-ceae-595e-8997-d05f6a7a8dfe",
                       "datadeps", "SMPL_models", "SMPL_MALE.npz")
default_out = joinpath(dirname(@__DIR__), "SMPL_MALE.smplbin")

npz_path = length(ARGS) >= 1 ? ARGS[1] : default_in
out_path = length(ARGS) >= 2 ? ARGS[2] : default_out

convert_npz_to_bin(npz_path, out_path)
