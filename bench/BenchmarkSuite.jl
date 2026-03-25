# BenchmarkSuite.jl — Developer benchmark definitions for SMPL.jl.
#
# Provides five public functions:
#   benchmark_pipeline(model, β, θ, trans; seconds)
#       → BenchResult for the full smpl_lbs forward pass (any backend).
#
#   benchmark_pipeline_gpu(gpu_model, β_gpu, θ_gpu, trans_gpu, sync_fn; seconds)
#       → BenchResult for the GPU forward pass, with device synchronization.
#
#   benchmark_components(model::BodyModel, β, θ; seconds)
#       → NamedTuple of BenchResult, one per LBS pipeline step (CPU only).
#
#   benchmark_static(exe_path, smplbin_path, β, θ, trans; n)
#       → NamedTuple{(:min_s, :median_s, :times)} timing the compiled static executable.
#
#   benchmark_static_julia(smplbin_path, N_b, N_j; seconds)
#       → BenchResult for the GC-free static_smpl_lbs pipeline called directly from Julia
#         (no subprocess, no exe required — needs only the .smplbin model file).
#
# Usage: include this file from bench/run_benchmarks.jl (not a registered package).
#
# Note: BenchmarkTools is intentionally not used here. Its $ interpolation syntax
# was broken by Julia 1.12's parser changes, and without it @benchmarkable cannot
# see local variables (it runs in module scope). A simple @elapsed loop is more
# reliable and sufficient for a developer profiling report.

module BenchmarkSuite

using SMPL
using LinearAlgebra: mul!, I
using StaticTools     # MallocMatrix, MallocVector — needed for benchmark_static_julia
using StaticArrays    # SVector — used by rodrigues inside static_smpl_lbs

# Path to repo root (bench/ is one level below the repo root)
const _REPO_ROOT = dirname(@__DIR__)

# Load the .smplbin binary reader (defines create_smpl → BodyModel{Float32, MallocMatrix})
# This file uses only `using StaticTools`; it does not redefine BodyModel.
include(joinpath(_REPO_ROOT, "static", "static_io.jl"))

export BenchResult, benchmark_pipeline, benchmark_pipeline_gpu,
       benchmark_components, benchmark_static, benchmark_static_julia


# ---------------------------------------------------------------------------
# BenchResult — lightweight timing result
# ---------------------------------------------------------------------------

"""
    BenchResult

Timing result from a benchmark run.

Fields:
- `min_ns`    — minimum observed time in nanoseconds
- `median_ns` — median observed time in nanoseconds
- `n_samples` — number of timed samples collected
"""
struct BenchResult
    min_ns    :: Float64
    median_ns :: Float64
    n_samples :: Int
end


# ---------------------------------------------------------------------------
# _time_fn — core timer (private)
# ---------------------------------------------------------------------------

"""
    _time_fn(f; seconds=2.0, min_samples=5) -> BenchResult

Time zero-argument callable `f` for at least `seconds` wall-clock time
(and at least `min_samples` iterations). Three warmup calls are made first
to ensure JIT compilation has occurred before measurement begins.
"""
function _time_fn(f::F; seconds::Real=2.0, min_samples::Int=5) where {F}
    # Warmup: force JIT compilation and cache warming
    for _ in 1:3
        f()
    end

    times = Float64[]
    t_deadline = time() + Float64(seconds)
    while time() < t_deadline || length(times) < min_samples
        push!(times, @elapsed(f()) * 1e9)   # nanoseconds
    end
    sort!(times)
    n = length(times)
    return BenchResult(times[1], times[n ÷ 2 + 1], n)
end


# ---------------------------------------------------------------------------
# benchmark_pipeline — full forward pass, any backend
# ---------------------------------------------------------------------------

"""
    benchmark_pipeline(model, β, θ, trans; seconds=5) -> BenchResult

Benchmark the full `smpl_lbs` forward pass. Works for any backend (CPU or GPU
— for GPU use `benchmark_pipeline_gpu` to add device synchronisation).

# Arguments
- `model`: loaded BodyModel or SUPRModel (CPU or GPU)
- `β`: shape coefficients  (N_b,)
- `θ`: axis-angle pose     (N_j*3,)
- `trans`: global translation (3,)
- `seconds`: target benchmark duration in seconds (default 5)
"""
function benchmark_pipeline(model, β, θ, trans; seconds=5)
    return _time_fn(seconds=seconds) do
        smpl_lbs(model, β, θ, trans)
    end
end


# ---------------------------------------------------------------------------
# benchmark_pipeline_gpu — full forward pass, GPU backend with sync
# ---------------------------------------------------------------------------

"""
    benchmark_pipeline_gpu(gpu_model, β_gpu, θ_gpu, trans_gpu, sync_fn; seconds=5)
                          -> BenchResult

Benchmark the GPU forward pass. `sync_fn` is a zero-argument callable that
blocks until all pending GPU work completes (typically `CUDA.synchronize`).
Without synchronisation the benchmark would measure kernel-launch latency only.

# Example
```julia
using CUDA, Adapt
gpu_model = Adapt.adapt(CuArray, model)
result = benchmark_pipeline_gpu(gpu_model, CuArray(β), CuArray(θ), CuArray(trans),
                                 CUDA.synchronize)
```
"""
function benchmark_pipeline_gpu(gpu_model, β_gpu, θ_gpu, trans_gpu, sync_fn; seconds=5)
    return _time_fn(seconds=seconds) do
        smpl_lbs(gpu_model, β_gpu, θ_gpu, trans_gpu)
        sync_fn()
    end
end


# ---------------------------------------------------------------------------
# benchmark_components — per-step breakdown, CPU BodyModel only
# ---------------------------------------------------------------------------

"""
    benchmark_components(model::BodyModel{ET}, β, θ; seconds=2)
                        -> NamedTuple of BenchResult

Benchmark each of the 8 LBS pipeline steps independently using a CPU BodyModel.
Each step is given the output of the previous step as a fixed input, so timings
are independent and additive.

Steps benchmarked:
  (1) shape_blend   — v_s = v̄ + S·β              matrix-vector multiply + reshape
  (2) joint_pos     — J = R_J · v_s               matmul
  (3) rodrigues     — R_k = rodrigues(θ_k)         N_j SMatrix evaluations
  (4) pose_features — ψ = vec(Rᵀ − I)             permutedims + broadcast
  (5) pose_blend    — v_p = v_s + P·ψ             matmul + reshape
  (6) fwd_kin       — G_k = FK(R, J, parents)     sequential kinematic chain
  (7) blend_xforms  — T_i = Σ w·G                 matmul + reshape
  (8) skinning      — v_i = T_i·[v_p_i;1]         per-vertex mul! loop

# Returns
NamedTuple with keys `shape_blend`, `joint_pos`, `rodrigues`, `pose_features`,
`pose_blend`, `fwd_kin`, `blend_xforms`, `skinning` — all `BenchResult`.
"""
function benchmark_components(
    model   :: SMPL.BodyModel{ET},
    β       :: AbstractVector{ET},
    θ       :: AbstractVector{ET};
    seconds :: Real = 2,
) where ET

    N_v = size(model.v_template, 1)   # number of vertices (e.g. 6890)
    N_j = size(model.J_regressor, 1)  # number of joints   (e.g. 24)
    N_b = length(β)                   # shape components used

    # ---- Precompute all pipeline intermediates once ----------------------
    # (1)
    v_shaped = model.v_template .+ reshape((@view model.shapedirs[:, 1:N_b]) * β, N_v, 3)
    # (2)
    J        = model.J_regressor * v_shaped
    # (3)
    θ_cpu    = Array(θ)
    θ_mat    = reshape(θ_cpu, 3, N_j)
    rot_mats = zeros(ET, 3, 3, N_j)
    @inbounds for k in axes(rot_mats, 3)
        rot_mats[:, :, k] .= SMPL.rodrigues(@view θ_mat[:, k])
    end
    # (4)
    I3   = Matrix{ET}(I, 3, 3)
    ψ    = reshape(permutedims(rot_mats[:, :, 2:end], (2, 1, 3)) .- I3, 1, :)
    # (5)
    v_posed = v_shaped .+ reshape(ψ * model.posedirs, N_v, 3)
    # (6)
    J_cpu   = Array(J)'
    _, A    = SMPL.forward_kinematics(rot_mats, J_cpu, model.parents)
    # (7)
    A_flat  = reshape(A, 16, N_j)
    T_blend = reshape(A_flat * model.lbs_weights', 4, 4, N_v)
    # (8)
    ones_row  = fill!(similar(model.v_template, 1, N_v), one(ET))
    v_posed_h = vcat(v_posed', ones_row)

    # Capture model fields as named locals for clarity in the closures below
    v_template  = model.v_template
    shapedirs   = model.shapedirs
    posedirs    = model.posedirs
    J_regressor = model.J_regressor
    lbs_weights = model.lbs_weights
    parents     = model.parents

    # ---- Benchmark each step via timed closure ---------------------------

    # (1) Shape blend shapes: v_s = v̄ + S·β
    t1 = _time_fn(seconds=seconds) do
        v_template .+ reshape((@view shapedirs[:, 1:N_b]) * β, N_v, 3)
    end

    # (2) Joint positions: J = R_J · v_s
    t2 = _time_fn(seconds=seconds) do
        J_regressor * v_shaped
    end

    # (3) Per-joint rodrigues: R_k = rodrigues(θ_k) for all k
    t3 = _time_fn(seconds=seconds) do
        rm = zeros(ET, 3, 3, N_j)
        @inbounds for k in axes(rm, 3)
            rm[:, :, k] .= SMPL.rodrigues(@view θ_mat[:, k])
        end
        rm
    end

    # (4) Pose feature vector: ψ = vec(Rᵀ − I) for joints 2..N_j
    t4 = _time_fn(seconds=seconds) do
        reshape(permutedims(rot_mats[:, :, 2:end], (2, 1, 3)) .- I3, 1, :)
    end

    # (5) Pose blend shapes: v_p = v_s + P·ψ
    t5 = _time_fn(seconds=seconds) do
        v_shaped .+ reshape(ψ * posedirs, N_v, 3)
    end

    # (6) Forward kinematics: G_k = G_{pa(k)} · T_k^local
    t6 = _time_fn(seconds=seconds) do
        SMPL.forward_kinematics(rot_mats, J_cpu, parents)
    end

    # (7) Blend transforms: T_i = Σ_k w_{ki}·A_k
    t7 = _time_fn(seconds=seconds) do
        reshape(A_flat * lbs_weights', 4, 4, N_v)
    end

    # (8) Linear blend skinning: v_i = T_i·[v_p_i; 1]
    t8 = _time_fn(seconds=seconds) do
        out = similar(v_posed_h)
        @inbounds for i in axes(out, 2)
            mul!(@view(out[:, i]), @view(T_blend[:, :, i]), @view(v_posed_h[:, i]))
        end
        out
    end

    return (
        shape_blend   = t1,
        joint_pos     = t2,
        rodrigues     = t3,
        pose_features = t4,
        pose_blend    = t5,
        fwd_kin       = t6,
        blend_xforms  = t7,
        skinning      = t8,
    )
end


# ---------------------------------------------------------------------------
# benchmark_static — time the compiled static executable end-to-end
# ---------------------------------------------------------------------------

"""
    benchmark_static(exe_path, smplbin_path, β, θ, trans; n=20)
                    -> NamedTuple{(:min_s, :median_s, :times), ...}

Time the static compiled executable over `n` subprocess calls. Measures the
full round-trip: write binary inputs → launch process → wait for completion.

Binary I/O format expected by the executable:
  Input:  UInt64 length (number of Float32 elements) + raw Float32 bytes
  Output: UInt64 rows + UInt64 cols + raw Float32 bytes (column-major)
"""
function benchmark_static(
    exe_path     :: AbstractString,
    smplbin_path :: AbstractString,
    β            :: AbstractVector{Float32},
    θ            :: AbstractVector{Float32},
    trans        :: AbstractVector{Float32};
    n            :: Int = 20,
)
    isfile(exe_path)     || error("Static exe not found: $exe_path\nRun `julia compile.jl` first.")
    isfile(smplbin_path) || error(".smplbin model not found: $smplbin_path")

    dir = mktempdir()
    _write_f32_bin(joinpath(dir, "betas.bin"),  β)
    _write_f32_bin(joinpath(dir, "poses.bin"),  θ)
    _write_f32_bin(joinpath(dir, "trans.bin"),  trans)
    verts_bin  = joinpath(dir, "verts.bin")
    joints_bin = joinpath(dir, "joints.bin")

    cmd = `$exe_path $smplbin_path $(joinpath(dir,"betas.bin")) $(joinpath(dir,"poses.bin")) $(joinpath(dir,"trans.bin")) $verts_bin $joints_bin`

    # Suppress exe stdout/stderr (it prints "Loading..." and "Done." per call)
    quiet = pipeline(cmd; stdout=devnull, stderr=devnull)
    run(quiet; wait=true)  # warmup

    times = Vector{Float64}(undef, n)
    for i in 1:n
        times[i] = @elapsed run(quiet; wait=true)
    end
    sort!(times)

    return (min_s=times[1], median_s=times[n ÷ 2], times=times)
end


# ---------------------------------------------------------------------------
# _write_f32_bin — write Float32 vector with UInt64 length prefix
# ---------------------------------------------------------------------------

function _write_f32_bin(path::String, v::AbstractVector{Float32})
    open(path, "w") do f
        write(f, UInt64(length(v)))
        write(f, collect(Float32, v))
    end
end

# ---------------------------------------------------------------------------
# Static compute functions — GC-free LBS pipeline (copied from staticSMPL.jl)
# These run with MallocMatrix inputs (no GC pressure, no BLAS).
# ---------------------------------------------------------------------------

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

# GC-free forward kinematics.
# rot_flat (9,N_j), J (N_j,3), parents (N_j,) → G_posed (16,N_j), A (16,N_j)
function _static_fk!(
    G_posed  :: MallocMatrix{Float32},
    A        :: MallocMatrix{Float32},
    rot_flat :: MallocMatrix{Float32},
    J        :: MallocMatrix{Float32},
    parents  :: Vector{Int32},
    N_j      :: Int,
)
    local_T = MallocMatrix{Float32}(undef, 16, N_j)
    fill!(local_T, 0f0)
    @inbounds for k in 1:N_j
        p = parents[k]
        for c in 1:3, r in 1:3
            local_T[r + 4*(c-1), k] = rot_flat[r + 3*(c-1), k]
        end
        for d in 1:3
            local_T[d + 12, k] = J[k, d] - (k > 1 ? J[p, d] : 0f0)
        end
        local_T[16, k] = 1f0
    end

    fill!(A, 0f0)
    @inbounds for i in 1:16
        A[i, 1] = local_T[i, 1]
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

    @inbounds for k in 1:N_j, i in 1:16
        G_posed[i, k] = A[i, k]
    end

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

# GC-free 8-step LBS forward pass using MallocMatrix inputs (no GC, no BLAS).
function _static_smpl_lbs(
    model :: SMPL.BodyModel{Float32, MallocMatrix{Float32}},
    β     :: MallocVector{Float32},
    θ     :: MallocVector{Float32},
    trans :: MallocVector{Float32},
)
    N_v = size(model.v_template, 1)
    N_j = size(model.J_regressor, 1)
    N_b = length(β)

    # (1) v_shaped = v_template + reshape(shapedirs[:,1:N_b] * β, N_v, 3)
    delta_v = MallocVector{Float32}(undef, N_v * 3)
    fill!(delta_v, 0f0)
    @inbounds for j in 1:N_b
        bj = β[j]
        for i in 1:N_v*3
            delta_v[i] += model.shapedirs[i, j] * bj
        end
    end
    v_shaped = MallocMatrix{Float32}(undef, N_v, 3)
    @inbounds for c in 1:3, i in 1:N_v
        v_shaped[i, c] = model.v_template[i, c] + delta_v[i + N_v*(c-1)]
    end

    # (2) J = J_regressor * v_shaped
    J = MallocMatrix{Float32}(undef, N_j, 3)
    fill!(J, 0f0)
    @inbounds for c in 1:3
        for j in 1:N_v
            for i in 1:N_j
                J[i, c] += model.J_regressor[i, j] * v_shaped[j, c]
            end
        end
    end

    # (3) rodrigues per joint → rot_flat (9, N_j)
    rot_flat = MallocMatrix{Float32}(undef, 9, N_j)
    @inbounds for k in 1:N_j
        R = SMPL.rodrigues(SVector{3,Float32}(θ[1+3*(k-1)], θ[2+3*(k-1)], θ[3+3*(k-1)]))
        for c in 1:3, r in 1:3
            rot_flat[r + 3*(c-1), k] = R[r, c]
        end
    end

    # (4) ψ = vec(R_{2..N_j}^T − I)
    psi = MallocVector{Float32}(undef, (N_j-1)*9)
    @inbounds for k in 2:N_j
        base = (k-2)*9
        for c in 1:3, r in 1:3
            psi[base + r + 3*(c-1)] = rot_flat[c + 3*(r-1), k] - (r == c ? 1f0 : 0f0)
        end
    end

    # (5) v_posed = v_shaped + reshape(ψ * posedirs, N_v, 3)
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

    # (6) Forward kinematics
    G_posed = MallocMatrix{Float32}(undef, 16, N_j)
    A       = MallocMatrix{Float32}(undef, 16, N_j)
    _static_fk!(G_posed, A, rot_flat, J, model.parents, N_j)

    # (7) T_blend = A * lbs_weights'  (16, N_v)
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

    # (8) Linear blend skinning: v_i = T_i · [v_p_i; 1]
    verts  = MallocMatrix{Float32}(undef, N_v, 3)
    joints = MallocMatrix{Float32}(undef, N_j, 3)
    @inbounds for n in 1:N_v
        for r in 1:3
            acc = T_blend[r + 12, n]
            for c in 1:3
                acc += T_blend[r + 4*(c-1), n] * v_posed[n, c]
            end
            verts[n, r] = acc + trans[r]
        end
    end
    @inbounds for k in 1:N_j
        for r in 1:3
            joints[k, r] = G_posed[r + 12, k] + trans[r]
        end
    end

    return verts, joints
end


# ---------------------------------------------------------------------------
# benchmark_static_julia — GC-free pipeline called directly from Julia
# ---------------------------------------------------------------------------

"""
    benchmark_static_julia(smplbin_path, N_b, N_j; seconds=5) -> BenchResult

Benchmark the GC-free LBS pipeline (`_static_smpl_lbs`) using
`MallocMatrix`-backed model loaded from `smplbin_path`. No subprocess, no
compiled exe required — only the `.smplbin` model file.

This isolates the algorithmic cost of the static implementation (no BLAS,
manual matrix multiply loops) from subprocess launch and model-load overhead.

# Arguments
- `smplbin_path`: path to a `.smplbin` model file (e.g. `SMPL_FEMALE.smplbin`)
- `N_b`: number of shape components to use (e.g. 10)
- `N_j`: number of joints (e.g. 24 for SMPL)
- `seconds`: target benchmark duration in seconds (default 5)
"""
function benchmark_static_julia(
    smplbin_path :: AbstractString,
    N_b          :: Int,
    N_j          :: Int;
    seconds      :: Real = 5,
)
    isfile(smplbin_path) || error("Static model not found: $smplbin_path\n" *
                                  "Run `julia scripts/convert_model.jl model.npz model.smplbin` first.")

    static_model = create_smpl(smplbin_path)

    β_m = MallocVector{Float32}(undef, N_b);   fill!(β_m, 0f0)
    θ_m = MallocVector{Float32}(undef, N_j*3); fill!(θ_m, 0f0)
    t_m = MallocVector{Float32}(undef, 3);     fill!(t_m, 0f0)

    return _time_fn(seconds=seconds) do
        _static_smpl_lbs(static_model, β_m, θ_m, t_m)
    end
end


end  # module BenchmarkSuite
