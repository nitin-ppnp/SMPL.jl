# staticSMPL.jl — JuliaC @main entrypoint for standalone SMPL executable.
#
# WHY this file exists separately from src/model.jl:
#   trim_mode = "unsafe" strips the Julia GC and IO dispatch table.
#   src/model.jl uses zeros(), ones(), vcat(), copy(), Array() — all GC-backed.
#   This file reimplements the full LBS pipeline using only MallocArray (malloc,
#   no GC) and ccall (no Julia IO dispatch), so every allocation survives trimming.
#
# Binary I/O mode (6 args):
#   smpl model.smplbin betas.bin poses.bin trans.bin verts_out.bin joints_out.bin
#   Input files:  UInt64 length-prefix + raw Float32 bytes
#   Output files: UInt64 rows + UInt64 cols + raw Float32 bytes (column-major)
#
# Interactive mode (1 arg):
#   smpl model.smplbin
#   Runs with zero betas/poses/trans and prints summary to stdout.

using LinearAlgebra
using StaticArrays

include("src/types.jl")
include("src/math.jl")        # rodrigues (SMatrix, stack-alloc), quat_feat
include("static/static_io.jl")  # create_smpl → BodyModel{Float32, MallocMatrix{Float32}}

const DEFAULT_MODEL_PATH = "C:\\Users\\nitin\\Desktop\\projects\\SMPL.jl\\SMPL_MALE.smplbin"

# Bypass Julia IO dispatch (stripped in unsafe mode) — call C puts directly.
function p(s::String)
    ccall(:puts, Cint, (Cstring,), s)
end


# ===========================================================================
# Binary I/O helpers (trimmer-safe ccall-based)
# ===========================================================================

# Read a 1D Float32 vector: UInt64 length-prefix + raw Float32 data
function _read_f32_vec(path::String) :: MallocVector{Float32}
    fp  = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), path, "rb")
    n   = Ref{UInt64}(0)
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), n, 8, 1, fp)
    arr = MallocVector{Float32}(undef, Int(n[]))
    ccall(:fread, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}),
          pointer(arr), sizeof(Float32), length(arr), fp)
    ccall(:fclose, Cint, (Ptr{Cvoid},), fp)
    return arr
end

# Write a 2D Float32 matrix: UInt64 rows + UInt64 cols + raw Float32 data (column-major)
function _write_f32_mat(path::String, mat::MallocMatrix{Float32})
    fp = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), path, "wb")
    r  = Ref{UInt64}(UInt64(size(mat, 1)))
    c  = Ref{UInt64}(UInt64(size(mat, 2)))
    ccall(:fwrite, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), r, 8, 1, fp)
    ccall(:fwrite, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}), c, 8, 1, fp)
    ccall(:fwrite, Csize_t, (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}),
          pointer(mat), sizeof(Float32), length(mat), fp)
    ccall(:fclose, Cint, (Ptr{Cvoid},), fp)
end


# ===========================================================================
# Static matrix arithmetic — no BLAS, no GC allocations
# All indices are 1-based (Julia convention).
# MallocMatrix stores data column-major, same as Julia Matrix.
# ===========================================================================

# y = A * x   (A: m×n, x: n-vec → y: m-vec)
function _mv!(y::MallocVector{Float32},
              A::MallocMatrix{Float32},
              x::MallocVector{Float32})
    m, n = size(A)
    fill!(y, 0f0)
    @inbounds for j in 1:n
        xj = x[j]
        for i in 1:m
            y[i] += A[i, j] * xj
        end
    end
end

# C = A * B   (A: m×k, B: k×n → C: m×n)
function _mm!(C::MallocMatrix{Float32},
              A::MallocMatrix{Float32},
              B::MallocMatrix{Float32})
    m, k = size(A)
    n    = size(B, 2)
    fill!(C, 0f0)
    @inbounds for j in 1:n
        for l in 1:k
            blj = B[l, j]
            for i in 1:m
                C[i, j] += A[i, l] * blj
            end
        end
    end
end


# ===========================================================================
# _static_fk! — GC-free forward kinematics
#
# Storage convention (avoids 3D MallocArray):
#   rot_flat  :: MallocMatrix{Float32}(9,  N_j)  — rot_flat[r+3*(c-1), k] = R_k[r,c]
#   J         :: MallocMatrix{Float32}(N_j, 3)   — J[k, d] = joint k position, coord d
#   G_posed   :: MallocMatrix{Float32}(16, N_j)  — G_posed[r+4*(c-1), k] = G_k[r,c]  (before offset removal)
#   A         :: MallocMatrix{Float32}(16, N_j)  — A[r+4*(c-1), k] = Â_k[r,c]  (LBS-ready)
# ===========================================================================

function _static_fk!(
    G_posed  :: MallocMatrix{Float32},   # (16, N_j) output: posed global transforms
    A        :: MallocMatrix{Float32},   # (16, N_j) output: LBS-ready transforms
    rot_flat :: MallocMatrix{Float32},   # (9,  N_j) input:  local rotations
    J        :: MallocMatrix{Float32},   # (N_j, 3)  input:  rest-pose joint positions
    parents  :: Vector{Int32},
    N_j      :: Int,
)
    # ---- Build local 4×4 transforms: T_k = [R_k | rel_t_k; 0ᵀ | 1] --------
    local_T = MallocMatrix{Float32}(undef, 16, N_j)
    fill!(local_T, 0f0)
    @inbounds for k in 1:N_j
        p = parents[k]
        # Rotation block (top-left 3×3): local_T[r+4*(c-1), k] = R_k[r,c]
        for c in 1:3, r in 1:3
            local_T[r + 4*(c-1), k] = rot_flat[r + 3*(c-1), k]
        end
        # Translation: rel_t = J[k,:] - J[parent[k],:]  (root relative to origin)
        for d in 1:3
            local_T[d + 12, k] = J[k, d] - (k > 1 ? J[p, d] : 0f0)
        end
        local_T[16, k] = 1f0   # homogeneous bottom-right
    end

    # ---- Accumulate global transforms: G_k = G_{pa(k)} * T_k^{local} -------
    fill!(A, 0f0)
    @inbounds for i in 1:16
        A[i, 1] = local_T[i, 1]   # root: G_1 = T_1^{local}
    end
    @inbounds for k in 2:N_j
        p = parents[k]
        for c in 1:4, r in 1:4
            acc = 0f0
            for m in 1:4
                acc += A[r + 4*(m-1), p] * local_T[m + 4*(c-1), k]
            end
            A[r + 4*(c-1), k] = acc
        end
    end

    # ---- Save posed transforms (G_posed = A before rest-pose removal) --------
    @inbounds for k in 1:N_j, i in 1:16
        G_posed[i, k] = A[i, k]
    end

    # ---- Subtract rest-pose joint offset: A_k[:,4] -= G_k[:,1:3] * J[k,:] ---
    # Implements A_k = G_k · [I | -J_k; 0ᵀ | 1]   (Eq.7, SMPL paper)
    @inbounds for k in 1:N_j
        for r in 1:3
            t = 0f0
            for d in 1:3
                t += A[r + 4*(d-1), k] * J[k, d]
            end
            A[r + 12, k] -= t
        end
    end
end


# ===========================================================================
# static_smpl_lbs — GC-free 8-step LBS forward pass for BodyModel (SMPL/SMPLX)
#
# Returns (verts::MallocMatrix{Float32}(N_v,3), joints::MallocMatrix{Float32}(N_j,3))
# All temporaries use MallocMatrix — no GC allocations anywhere.
# ===========================================================================

function static_smpl_lbs(
    model :: BodyModel{Float32, MallocMatrix{Float32}},
    β     :: MallocVector{Float32},
    θ     :: MallocVector{Float32},
    trans :: MallocVector{Float32},
) :: Tuple{MallocMatrix{Float32}, MallocMatrix{Float32}}

    N_v = size(model.v_template, 1)
    N_j = size(model.J_regressor, 1)
    N_b = length(β)

    # ------------------------------------------------------------------
    # (1) v_shaped = v_template + reshape(shapedirs[:, 1:N_b] * β, N_v, 3)
    #     shapedirs is (N_v*3, N_b); β is (N_b,) → delta_v is (N_v*3,)
    # ------------------------------------------------------------------
    delta_v = MallocVector{Float32}(undef, N_v * 3)
    fill!(delta_v, 0f0)
    @inbounds for j in 1:N_b
        bj = β[j]
        for i in 1:N_v*3
            delta_v[i] += model.shapedirs[i, j] * bj
        end
    end

    # delta_v is flat (N_v*3,) in column-major reshape order:
    #   delta_v[i + N_v*(c-1)] = coord c offset for vertex i
    v_shaped = MallocMatrix{Float32}(undef, N_v, 3)
    @inbounds for c in 1:3, i in 1:N_v
        v_shaped[i, c] = model.v_template[i, c] + delta_v[i + N_v*(c-1)]
    end

    # ------------------------------------------------------------------
    # (2) J = J_regressor * v_shaped   (N_j × N_v) × (N_v × 3) → (N_j, 3)
    # ------------------------------------------------------------------
    J = MallocMatrix{Float32}(undef, N_j, 3)
    fill!(J, 0f0)
    @inbounds for c in 1:3
        for j in 1:N_v
            @inbounds for i in 1:N_j
                J[i, c] += model.J_regressor[i, j] * v_shaped[j, c]
            end
        end
    end

    # ------------------------------------------------------------------
    # (3) rodrigues(θ_k) per joint → rot_flat (9, N_j)
    #     rot_flat[r + 3*(c-1), k] = R_k[r,c]  (column-major 3×3)
    # ------------------------------------------------------------------
    rot_flat = MallocMatrix{Float32}(undef, 9, N_j)
    @inbounds for k in 1:N_j
        R = rodrigues(SVector{3,Float32}(θ[1 + 3*(k-1)], θ[2 + 3*(k-1)], θ[3 + 3*(k-1)]))
        for c in 1:3, r in 1:3
            rot_flat[r + 3*(c-1), k] = R[r, c]
        end
    end

    # ------------------------------------------------------------------
    # (4) ψ = vec(R_{2..N_j}^T − I)   shape: ((N_j-1)*9,)
    #     Matches: reshape(permutedims(rot_mats[:,:,2:end],(2,1,3)) .- I3, 1, :)
    #     ψ[r + 3*(c-1) + 9*(k-2)] = R_k^T[r,c] - δ_{rc}
    #                               = R_k[c,r] - δ_{rc}
    #                               = rot_flat[c + 3*(r-1), k] - δ_{rc}
    # ------------------------------------------------------------------
    psi = MallocVector{Float32}(undef, (N_j-1)*9)
    @inbounds for k in 2:N_j
        base = (k-2)*9
        for c in 1:3, r in 1:3
            psi[base + r + 3*(c-1)] = rot_flat[c + 3*(r-1), k] - (r == c ? 1f0 : 0f0)
        end
    end

    # ------------------------------------------------------------------
    # (5) v_posed = v_shaped + reshape(ψ * posedirs, N_v, 3)
    #     posedirs is ((N_j-1)*9, N_v*3); ψ·posedirs → (N_v*3,) row-vec collapsed
    #     delta_p[j] = Σ_i psi[i] * posedirs[i, j]
    # ------------------------------------------------------------------
    delta_p = MallocVector{Float32}(undef, N_v*3)
    fill!(delta_p, 0f0)
    @inbounds for j in 1:N_v*3
        acc = 0f0
        for i in 1:(N_j-1)*9
            acc += model.posedirs[i, j] * psi[i]
        end
        delta_p[j] = acc
    end

    v_posed = MallocMatrix{Float32}(undef, N_v, 3)
    @inbounds for c in 1:3, i in 1:N_v
        v_posed[i, c] = v_shaped[i, c] + delta_p[i + N_v*(c-1)]
    end

    # ------------------------------------------------------------------
    # (6) G_posed, A = forward_kinematics(rot, J, parents)
    #     Both stored as (16, N_j) flat:  mat[r + 4*(c-1), k] = mat[r,c,k]
    # ------------------------------------------------------------------
    G_posed = MallocMatrix{Float32}(undef, 16, N_j)
    A       = MallocMatrix{Float32}(undef, 16, N_j)
    _static_fk!(G_posed, A, rot_flat, J, model.parents, N_j)

    # ------------------------------------------------------------------
    # (7) T_blend = reshape(A * lbs_weights', 4, 4, N_v)
    #     A (16,N_j) × lbs_weights' (N_j,N_v) → T_blend (16,N_v)
    #     lbs_weights is (N_v, N_j), so lbs_weights'[k,j] = lbs_weights[j,k]
    # ------------------------------------------------------------------
    T_blend = MallocMatrix{Float32}(undef, 16, N_v)
    fill!(T_blend, 0f0)
    @inbounds for n in 1:N_v
        for k in 1:N_j
            w = model.lbs_weights[n, k]
            for i in 1:16
                T_blend[i, n] += A[i, k] * w
            end
        end
    end

    # ------------------------------------------------------------------
    # (8) v_i = T_i · [v_p_i; 1]   — linear blend skinning
    #     T_blend[r + 4*(c-1), n] = T_n[r,c]
    #     v_h[r, n] = Σ_{c=1}^{3} T_n[r,c] * v_posed[n,c] + T_n[r,4] * 1
    #     verts[n, r] = v_h[r, n] + trans[r]   (for r = 1,2,3)
    # ------------------------------------------------------------------
    verts  = MallocMatrix{Float32}(undef, N_v, 3)
    joints = MallocMatrix{Float32}(undef, N_j, 3)

    @inbounds for n in 1:N_v
        for r in 1:3
            acc = T_blend[r + 12, n]   # T_n[r,4] * 1 (homogeneous translation)
            for c in 1:3
                acc += T_blend[r + 4*(c-1), n] * v_posed[n, c]
            end
            verts[n, r] = acc + trans[r]
        end
    end

    # Extract joint world positions: G_posed[r, 4, k] = G_posed[r+12, k]
    @inbounds for k in 1:N_j
        for r in 1:3
            joints[k, r] = G_posed[r + 12, k] + trans[r]
        end
    end

    return verts, joints
end


# ===========================================================================
# @main — executable entry point
# ===========================================================================

function @main(args::Vector{String})::Cint
    if length(args) < 1
        p("Usage: smpl model.smplbin [betas.bin poses.bin trans.bin verts_out.bin joints_out.bin]")
        p("  6-arg mode: read inputs from binary files, write outputs to binary files")
        p("  1-arg mode: run with zero inputs, print summary to stdout")
        return 1
    end

    model_path = args[1]
    if !isfile(model_path)
        p(string("Error: model file not found: ", model_path))
        return 1
    end

    p(string("Loading: ", model_path))
    smpl = create_smpl(model_path)

    if length(args) >= 6
        # ------------------------------------------------------------------
        # Binary I/O mode: read betas/poses/trans from files, write outputs
        # ------------------------------------------------------------------
        betas = _read_f32_vec(args[2])
        poses = _read_f32_vec(args[3])
        trans = _read_f32_vec(args[4])

        verts, joints = static_smpl_lbs(smpl, betas, poses, trans)

        _write_f32_mat(args[5], verts)
        _write_f32_mat(args[6], joints)
        p("Done.")
        return 0

    else
        # ------------------------------------------------------------------
        # Interactive mode: zero inputs, print summary
        # ------------------------------------------------------------------
        betas = MallocVector{Float32}(undef, 10); fill!(betas, 0f0)
        poses = MallocVector{Float32}(undef, 72); fill!(poses, 0f0)
        trans = MallocVector{Float32}(undef,  3); fill!(trans, 0f0)

        verts, joints = static_smpl_lbs(smpl, betas, poses, trans)

        p("=== SMPL Forward Pass ===")
        p(string("Vertices : ", size(verts,  1), " x 3"))
        p(string("Joints   : ", size(joints, 1), " x 3"))

        p("--- First 5 vertices (x, y, z) ---")
        @inbounds for i in 1:5
            p(string("  v[", i, "]: (",
                     round(verts[i,1]; digits=4), ", ",
                     round(verts[i,2]; digits=4), ", ",
                     round(verts[i,3]; digits=4), ")"))
        end

        xmin = xmax = verts[1,1]
        ymin = ymax = verts[1,2]
        zmin = zmax = verts[1,3]
        @inbounds for i in 2:size(verts,1)
            verts[i,1] < xmin && (xmin = verts[i,1])
            verts[i,1] > xmax && (xmax = verts[i,1])
            verts[i,2] < ymin && (ymin = verts[i,2])
            verts[i,2] > ymax && (ymax = verts[i,2])
            verts[i,3] < zmin && (zmin = verts[i,3])
            verts[i,3] > zmax && (zmax = verts[i,3])
        end
        p("--- Vertex bounding box ---")
        p(string("  X: [", round(xmin; digits=4), ", ", round(xmax; digits=4), "]"))
        p(string("  Y: [", round(ymin; digits=4), ", ", round(ymax; digits=4), "]"))
        p(string("  Z: [", round(zmin; digits=4), ", ", round(zmax; digits=4), "]"))

        return 0
    end
end
