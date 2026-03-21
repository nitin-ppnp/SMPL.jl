# static/static_io.jl — Binary .smplbin model loader for JuliaC static compilation.
#
# This file is NOT included in the main SMPL.jl package. It is included only by
# staticSMPL.jl (the JuliaC @main entrypoint) and compiled via compile.jl.
#
# Why a separate file?
#   The JuliaC trimmer (trim_mode = "unsafe") strips the Julia runtime's GC and
#   IO dispatch table. Standard Julia IO (NPZ.jl, open/read) is not available.
#   This file uses ccall(:fread, ...) and ccall(:fopen, ...) directly — pure C
#   calls with no Julia IO dispatch, fully trimmer-safe.
#
# The BodyModel struct from src/types.jl is reused directly. With StaticTools.jl,
# the concrete instantiation BodyModel{Float32, MallocMatrix{Float32}} is used.
# MallocMatrix allocates via malloc (not the GC), which is available in trimmer mode.
#
# Binary format (.smplbin):
#   Header: "SMPL" magic (4 bytes) + version UInt32 (4 bytes) = 8 bytes total
#   Each array: UInt8 ndims | UInt64×ndims shape | UInt8 dtype | raw data bytes
#   Arrays in order: v_template, shapedirs, posedirs, J_regressor, parents,
#                    lbs_weights, faces
#   Convert NPZ → .smplbin with:  julia scripts/convert_model.jl model.npz out.smplbin

using StaticTools


# ---------------------------------------------------------------------------
# Internal binary read helpers
# ---------------------------------------------------------------------------

# Read a 2D Float32 matrix from the open file pointer.
# Reads: UInt8 ndims (expected 2), two UInt64 shape values, UInt8 dtype, then raw f32 data.
function _fread_f32_mat(fp::Ptr{Cvoid}) :: MallocMatrix{Float32}
    skip  = Ref{UInt8}(0)
    r_ref = Ref{UInt64}(0)
    c_ref = Ref{UInt64}(0)

    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), skip,  1, 1, fp)  # ndims
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), r_ref, 8, 1, fp)  # rows
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), c_ref, 8, 1, fp)  # cols
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), skip,  1, 1, fp)  # dtype

    arr = MallocMatrix{Float32}(undef, Int(r_ref[]), Int(c_ref[]))
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}),
          pointer(arr), sizeof(Float32), length(arr), fp)
    return arr
end

# Read a 2D UInt32 matrix from the open file pointer.
function _fread_u32_mat(fp::Ptr{Cvoid}) :: MallocMatrix{UInt32}
    skip  = Ref{UInt8}(0)
    r_ref = Ref{UInt64}(0)
    c_ref = Ref{UInt64}(0)

    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), skip,  1, 1, fp)
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), r_ref, 8, 1, fp)
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), c_ref, 8, 1, fp)
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), skip,  1, 1, fp)

    arr = MallocMatrix{UInt32}(undef, Int(r_ref[]), Int(c_ref[]))
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}),
          pointer(arr), sizeof(UInt32), length(arr), fp)
    return arr
end

# Read a 1D Int32 vector from the open file pointer.
# Note: parents are stored as Int32 (1-indexed) in the .smplbin format.
function _fread_i32_vec(fp::Ptr{Cvoid}) :: MallocVector{Int32}
    skip  = Ref{UInt8}(0)
    n_ref = Ref{UInt64}(0)

    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), skip,  1, 1, fp)  # ndims (1)
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), n_ref, 8, 1, fp)  # length
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), skip,  1, 1, fp)  # dtype

    arr = MallocVector{Int32}(undef, Int(n_ref[]))
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}),
          pointer(arr), sizeof(Int32), length(arr), fp)
    return arr
end


# ---------------------------------------------------------------------------
# create_smpl — load BodyModel from .smplbin (trimmer-safe)
# ---------------------------------------------------------------------------

"""
    create_smpl(model_path::String) -> BodyModel{Float32, MallocMatrix{Float32}}

Load a pre-converted .smplbin model file and return a MallocMatrix-backed BodyModel.

This is the JuliaC trimmer-safe loader. It uses only C stdlib calls (fopen/fread/fclose)
and MallocMatrix allocation — no GC, no Julia IO dispatch.

Convert a .npz model to .smplbin with:
  julia scripts/convert_model.jl SMPL_MALE.npz SMPL_MALE.smplbin
"""
function create_smpl(model_path::String) :: BodyModel{Float32, MallocMatrix{Float32}}
    fp = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), model_path, "rb")

    # Skip 8-byte header: "SMPL" magic (4 bytes) + version UInt32 (4 bytes)
    header = Ref{UInt64}(0)
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), header, 8, 1, fp)

    # Read arrays in the order written by scripts/convert_model.jl
    v_template  = _fread_f32_mat(fp)   # (N_v, 3)
    shapedirs   = _fread_f32_mat(fp)   # (N_v*3, N_b)
    posedirs    = _fread_f32_mat(fp)   # ((N_j-1)*9, N_v*3)
    J_regressor = _fread_f32_mat(fp)   # (N_j, N_v)
    parents     = _fread_i32_vec(fp)   # (N_j,) Int32 1-indexed
    lbs_weights = _fread_f32_mat(fp)   # (N_v, N_j)
    faces       = _fread_u32_mat(fp)   # (N_f, 3) UInt32 1-indexed

    ccall(:fclose, Cint, (Ptr{Cvoid},), fp)

    # MallocMatrix is not compatible with Vector{Int32} for parents.
    # Convert the MallocVector to a plain Vector for the BodyModel.parents field.
    # This is a small O(N_j) copy — acceptable in the static path.
    parents_vec = Vector{Int32}(undef, length(parents))
    for i in eachindex(parents_vec)
        parents_vec[i] = parents[i]
    end

    # faces is stored as MallocMatrix but BodyModel.faces is Matrix{UInt32}.
    faces_mat = Matrix{UInt32}(undef, size(faces)...)
    for j in axes(faces, 2), i in axes(faces, 1)
        faces_mat[i, j] = faces[i, j]
    end

    return BodyModel{Float32, MallocMatrix{Float32}}(
        v_template, shapedirs, posedirs, J_regressor, lbs_weights,
        parents_vec, faces_mat,
    )
end
