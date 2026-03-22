# test/test_static_io.jl — End-to-end static binary format test.
#
# Tests the full pipeline: convert_model.jl → static_io.jl → smpl_lbs vs reference.
# Does NOT require JuliaC. Verifies that:
#   1. convert_npz_to_bin (the real converter) produces a valid .smplbin from the NPZ
#   2. The ccall-based reader (static/static_io.jl) loads it back correctly
#   3. smpl_lbs on the loaded model matches Python reference outputs at 1e-5 tolerance
#
# Run standalone:   julia --project=. test/test_static_io.jl
# Run via Pkg.test: included from runtests.jl when SMPL_TEST_STATIC != "false"

using SMPL
using Test
using StaticTools
using NPZ
using DataDeps

include(joinpath(@__DIR__, "..", "scripts", "convert_model.jl"))
include(joinpath(@__DIR__, "..", "static", "static_io.jl"))


@testset "Static binary: convert → load → forward pass" begin
    # Resolve the downloaded SMPL FEMALE NPZ via DataDeps (same path used by create_smpl_female)
    npz_path = joinpath(datadep"SMPL_models", "SMPL_FEMALE.npz")
    tmp      = tempname() * ".smplbin"

    try
        # Step 1: run the real converter (scripts/convert_model.jl)
        convert_npz_to_bin(npz_path, tmp)
        @test isfile(tmp)
        @test filesize(tmp) > 0

        # Step 2: load via the ccall-based static reader → MallocMatrix-backed model
        smpl_s = create_smpl(tmp)

        @testset "Array shapes" begin
            ref_model = create_smpl_female()
            @test size(smpl_s.v_template)  == size(ref_model.v_template)
            @test size(smpl_s.shapedirs)   == size(ref_model.shapedirs)
            @test size(smpl_s.posedirs)    == size(ref_model.posedirs)
            @test size(smpl_s.J_regressor) == size(ref_model.J_regressor)
            @test size(smpl_s.lbs_weights) == size(ref_model.lbs_weights)
            @test size(smpl_s.faces)       == size(ref_model.faces)
            @test length(smpl_s.parents)   == length(ref_model.parents)
        end

        @testset "parents sentinel" begin
            @test smpl_s.parents[1] == Int32(1)
            N_j = length(smpl_s.parents)
            @test all(1 .<= smpl_s.parents .<= N_j)
        end

        @testset "Forward pass vs Python reference" begin
            # MallocMatrix doesn't support BLAS elsize outside a static build —
            # materialise to plain Matrix. The shapes/values tests above already
            # confirm the loaded data is correct; this test verifies the full
            # convert → load → inference pipeline against Python reference outputs.
            smpl_cpu = BodyModel{Float32, Matrix{Float32}}(
                Matrix(smpl_s.v_template),
                Matrix(smpl_s.shapedirs),
                Matrix(smpl_s.posedirs),
                Matrix(smpl_s.J_regressor),
                Matrix(smpl_s.lbs_weights),
                smpl_s.parents,
                smpl_s.faces,
            )
            ref = npzread(joinpath(@__DIR__, "smpltest.npz"))
            out = smpl_lbs(smpl_cpu, ref["betas"], ref["poses"], ref["trans"])
            @test maximum(abs.(out.vertices .- ref["out_vertices"])) < 1e-5
            @test maximum(abs.(out.joints   .- ref["out_joints"]))   < 1e-5
        end

    finally
        isfile(tmp) && rm(tmp)
    end
end
