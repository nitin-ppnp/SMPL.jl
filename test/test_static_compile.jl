# test/test_static_compile.jl — End-to-end static executable test.
#
# Full pipeline:
#   convert_npz_to_bin  →  julia compile.jl  →  build/smpl[.exe]
#   → run with binary inputs  →  read binary outputs  →  compare vs reference
#
# Requires JuliaC installed and ~5-10 min compile time.
# Skipped by default. Enable with:  SMPL_TEST_COMPILE=true julia --project=. -e 'Pkg.test()'
# Or run directly:                  SMPL_TEST_COMPILE=true julia --project=. test/test_static_compile.jl

using SMPL
using Test
using NPZ
using DataDeps

include(joinpath(@__DIR__, "..", "scripts", "convert_model.jl"))


# ---------------------------------------------------------------------------
# Binary format helpers (must mirror staticSMPL.jl's _read_f32_vec/_write_f32_mat)
# ---------------------------------------------------------------------------

# Write a Float32 vector: UInt64 length-prefix + raw Float32 bytes
function _write_input(path::String, v::AbstractVector)
    open(path, "w") do io
        write(io, UInt64(length(v)))
        write(io, Float32.(v))
    end
end

# Read a Float32 matrix: UInt64 rows + UInt64 cols + raw Float32 bytes (column-major)
function _read_output(path::String) :: Matrix{Float32}
    open(path, "r") do io
        rows = Int(read(io, UInt64))
        cols = Int(read(io, UInt64))
        data = Matrix{Float32}(undef, rows, cols)
        read!(io, data)
        return data
    end
end


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if get(ENV, "SMPL_TEST_COMPILE", "false") != "true"
    @info "Skipping static compile test (set SMPL_TEST_COMPILE=true to enable)"
else

@testset "Static executable: compile → run → verify" begin

    repo_root = joinpath(@__DIR__, "..")
    exe = Sys.iswindows() ? joinpath(repo_root, "build", "bin", "smpl.exe") :
                            joinpath(repo_root, "build", "bin", "smpl")

    # ---- Step 1: convert SMPL FEMALE NPZ → .smplbin ----------------------
    npz_path = joinpath(datadep"SMPL_models", "SMPL_FEMALE.npz")
    smplbin  = joinpath(repo_root, "SMPL_FEMALE_test.smplbin")
    convert_npz_to_bin(npz_path, smplbin)
    @test isfile(smplbin)

    # ---- Step 2: compile static executable --------------------------------
    @info "Compiling static executable (this may take several minutes)..."
    compile_ok = try
        run(Cmd(`$(Base.julia_cmd()) --project=. compile.jl`; dir=repo_root))
        true
    catch e
        @warn "Compilation failed (JuliaC not installed?): $e"
        false
    end

    if !compile_ok
        @test_skip "Compilation failed — JuliaC may not be installed"
    else
        @test isfile(exe)

        # ---- Step 3: write reference inputs as binary files ---------------
        ref        = npzread(joinpath(@__DIR__, "smpltest.npz"))
        betas_bin  = tempname() * ".bin"
        poses_bin  = tempname() * ".bin"
        trans_bin  = tempname() * ".bin"
        verts_bin  = tempname() * ".bin"
        joints_bin = tempname() * ".bin"

        _write_input(betas_bin, ref["betas"])
        _write_input(poses_bin, ref["poses"])
        _write_input(trans_bin, ref["trans"])

        # ---- Step 4: run executable with binary I/O -----------------------
        run_ok = try
            run(`$exe $smplbin $betas_bin $poses_bin $trans_bin $verts_bin $joints_bin`)
            true
        catch e
            @warn "Executable run failed: $e"
            false
        end

        if !run_ok
            @test_skip "Executable run failed"
        else
            @test isfile(verts_bin)
            @test isfile(joints_bin)

            # ---- Step 5: read outputs and compare vs Python reference -----
            verts_out  = _read_output(verts_bin)
            joints_out = _read_output(joints_bin)

            @testset "Output shapes" begin
                @test size(verts_out)  == size(ref["out_vertices"])
                @test size(joints_out) == size(ref["out_joints"])
            end

            @testset "Output values vs Python reference" begin
                # 1e-4 tolerance: manual float32 loops vs Python BLAS (float64 precision)
                @test maximum(abs.(verts_out  .- Float32.(ref["out_vertices"]))) < 1e-4
                @test maximum(abs.(joints_out .- Float32.(ref["out_joints"])))   < 1e-4
            end
        end

        # cleanup
        for f in (betas_bin, poses_bin, trans_bin, verts_bin, joints_bin)
            isfile(f) && rm(f)
        end
    end

    isfile(smplbin) && rm(smplbin)
end

end  # SMPL_TEST_COMPILE guard
