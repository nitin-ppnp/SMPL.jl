using SMPL
using Test
using DataDeps
using NPZ

# Tolerance for numerical comparison against Python reference outputs.
const ATOL = 1e-5

function test_smpl()
    datapath = joinpath(@__DIR__, "smpltest.npz")
    data     = npzread(datapath)
    smpl     = create_smpl_female()

    betas = data["betas"]
    poses = data["poses"]
    trans = data["trans"]

    # Reference outputs from Python SMPL (stored as (N_v,3) and (N_j,3) in NPZ)
    out_vertices = data["out_vertices"]   # (N_v, 3)
    out_joints   = data["out_joints"]     # (N_j, 3)

    out = smpl_lbs(smpl, betas, poses, trans)

    # out.vertices is (N_v, 3); out_vertices from NPZ is (N_v, 3) — no transpose needed.
    # (The original test compared (3,N_v) vs (N_v,3)' — same values, different layout.)
    verts_ok  = maximum(abs.(out.vertices  .- out_vertices)) < ATOL
    joints_ok = maximum(abs.(out.joints    .- out_joints))   < ATOL

    return verts_ok && joints_ok
end

function test_smplx()
    datapath = joinpath(@__DIR__, "smplxtest.npz")
    data     = npzread(datapath)
    smplx    = create_smplx_neutral()

    betas = data["betas"]
    poses = data["poses"]
    trans = data["trans"]

    out_vertices = data["out_vertices"]   # (N_v, 3)
    out_joints   = data["out_joints"]     # (N_j, 3)

    out = smpl_lbs(smplx, betas, poses, trans)

    verts_ok  = maximum(abs.(out.vertices  .- out_vertices)) < ATOL
    joints_ok = maximum(abs.(out.joints    .- out_joints))   < ATOL

    return verts_ok && joints_ok
end

function test_supr()
    datapath = joinpath(@__DIR__, "suprtest.npz")
    data     = npzread(datapath)
    supr     = create_supr_neutral()

    betas = data["betas"]
    poses = data["poses"]
    trans = data["trans"]

    out_vertices = data["out_vertices"]   # (N_v, 3)
    out_joints   = data["out_joints"]     # (N_j, 3)

    out = smpl_lbs(supr, betas, poses, trans)

    verts_ok  = maximum(abs.(out.vertices  .- out_vertices)) < ATOL
    joints_ok = maximum(abs.(out.joints    .- out_joints))   < ATOL

    return verts_ok && joints_ok
end

# Wrap everything in one outer testset so a failure in one group
# (e.g. SUPR model not downloaded) does not abort the remaining tests.
@testset "SMPL.jl" begin

    @testset "SMPL" begin
        @test test_smpl()
    end

    @testset "SMPLX" begin
        @test test_smplx()
    end

    @testset "SUPR" begin
        @test test_supr()
    end

    if get(ENV, "SMPL_TEST_STATIC", "true") != "false"
        include("test_static_io.jl")
    end

    include("test_static_compile.jl")

    if get(ENV, "SMPL_TEST_GPU", "false") == "true"
        include("test_gpu.jl")
    end

end
