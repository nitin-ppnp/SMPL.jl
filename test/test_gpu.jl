# test_gpu.jl — GPU (CUDA) tests for SMPL.jl
#
# Gated on SMPL_TEST_GPU=true environment variable so that normal `Pkg.test()`
# skips GPU tests on machines without a CUDA GPU.
#
# To run:
#   SMPL_TEST_GPU=true julia --project=. -e 'using Pkg; Pkg.test()'
#
# Requirements:
#   CUDA.jl and Adapt.jl must be installed in the active environment, e.g.:
#   julia --project=. -e 'using Pkg; Pkg.add(["CUDA", "Adapt"])'

# Tolerance for GPU vs CPU comparison. Slightly relaxed from CPU-vs-Python
# 1e-5 to allow for Float32 rounding differences between CPU BLAS and GPU kernels.
const GPU_ATOL = 1e-4

# Attempt to load CUDA and Adapt; skip gracefully if unavailable.
cuda_available = false
try
    @eval using CUDA
    @eval using Adapt
    if CUDA.functional()
        global cuda_available = true
    else
        @info "CUDA.jl loaded but no functional GPU found — skipping GPU tests"
    end
catch e
    @info "Could not load CUDA.jl ($e) — skipping GPU tests"
end

if cuda_available

    # ------------------------------------------------------------------
    # Helper: verify that all AbstractMatrix fields of a BodyModel are
    # CuMatrix, while parents and faces remain CPU arrays.
    # ------------------------------------------------------------------
    function _check_bodymmodel_on_gpu(model_gpu)
        model_gpu.v_template  isa CuMatrix{Float32} &&
        model_gpu.shapedirs   isa CuMatrix{Float32} &&
        model_gpu.posedirs    isa CuMatrix{Float32} &&
        model_gpu.J_regressor isa CuMatrix{Float32} &&
        model_gpu.lbs_weights isa CuMatrix{Float32} &&
        model_gpu.parents     isa Vector{Int32}      &&   # stays on CPU
        model_gpu.faces       isa Matrix{UInt32}          # stays on CPU
    end

    @testset "GPU (CUDA)" begin

        # --------------------------------------------------------------
        # Test 1: Adapt.adapt correctly moves BodyModel matrices to GPU
        # --------------------------------------------------------------
        @testset "Adapt BodyModel to GPU" begin
            model     = create_smpl_female()
            model_gpu = Adapt.adapt(CuArray, model)

            @test _check_bodymmodel_on_gpu(model_gpu)
            # parents and faces are the same objects (no copy)
            @test model_gpu.parents === model.parents
            @test model_gpu.faces   === model.faces
        end

        # --------------------------------------------------------------
        # Test 2: GPU SMPL forward pass matches CPU within GPU_ATOL
        # --------------------------------------------------------------
        @testset "SMPL LBS on GPU" begin
            datapath = joinpath(@__DIR__, "smpltest.npz")
            data     = npzread(datapath)

            model_cpu = create_smpl_female()
            model_gpu = Adapt.adapt(CuArray, model_cpu)

            betas = data["betas"]
            poses = data["poses"]
            trans = data["trans"]

            # CPU forward pass
            out_cpu = smpl_lbs(model_cpu, betas, poses, trans)

            # GPU forward pass
            out_gpu = smpl_lbs(
                model_gpu,
                CuArray(betas),
                CuArray(poses),
                CuArray(trans),
            )

            # Output vertex/joint arrays should live on GPU
            @test out_gpu.vertices  isa CuMatrix{Float32}
            @test out_gpu.joints    isa CuMatrix{Float32}
            @test out_gpu.v_shaped  isa CuMatrix{Float32}
            @test out_gpu.v_posed   isa CuMatrix{Float32}
            # J_transforms is always CPU (sequential FK result)
            @test out_gpu.J_transforms isa Array{Float32, 3}
            # faces is the shared reference from the model
            @test out_gpu.faces === model_cpu.faces

            # Numerical agreement with CPU
            @test maximum(abs.(Array(out_gpu.vertices) .- out_cpu.vertices)) < GPU_ATOL
            @test maximum(abs.(Array(out_gpu.joints)   .- out_cpu.joints))   < GPU_ATOL
        end

        # --------------------------------------------------------------
        # Test 3: GPU SMPLX forward pass matches CPU within GPU_ATOL
        # --------------------------------------------------------------
        @testset "SMPLX LBS on GPU" begin
            datapath = joinpath(@__DIR__, "smplxtest.npz")
            data     = npzread(datapath)

            model_cpu = create_smplx_neutral()
            model_gpu = Adapt.adapt(CuArray, model_cpu)

            betas = data["betas"]
            poses = data["poses"]
            trans = data["trans"]

            out_cpu = smpl_lbs(model_cpu, betas, poses, trans)
            out_gpu = smpl_lbs(
                model_gpu,
                CuArray(betas),
                CuArray(poses),
                CuArray(trans),
            )

            # Output arrays on GPU
            @test out_gpu.vertices    isa CuMatrix{Float32}
            @test out_gpu.joints      isa CuMatrix{Float32}
            @test out_gpu.v_shaped    isa CuMatrix{Float32}
            @test out_gpu.v_posed     isa CuMatrix{Float32}
            # FK result and faces always on CPU
            @test out_gpu.J_transforms isa Array{Float32, 3}
            @test out_gpu.faces        === model_cpu.faces
            # Numerical agreement with CPU
            @test maximum(abs.(Array(out_gpu.vertices) .- out_cpu.vertices)) < GPU_ATOL
            @test maximum(abs.(Array(out_gpu.joints)   .- out_cpu.joints))   < GPU_ATOL
        end

        # --------------------------------------------------------------
        # Test 4: GPU SUPR forward pass matches CPU (model download required)
        # Skipped automatically when the SUPR model is not downloaded or is
        # invalid (e.g. credentials expired and the file contains an HTML error
        # page instead of a real NPZ).
        # --------------------------------------------------------------
        @testset "SUPR LBS on GPU" begin
            supr_available = false
            model_cpu = nothing
            try
                model_cpu      = create_supr_neutral()
                supr_available = true
            catch e
                @info "Skipping SUPR GPU test: could not load SUPR model ($e)"
            end

            if supr_available
                model_gpu = Adapt.adapt(CuArray, model_cpu)

                datapath = joinpath(@__DIR__, "suprtest.npz")
                data     = npzread(datapath)

                betas = data["betas"]
                poses = data["poses"]
                trans = data["trans"]

                # SUPRModel: verify J_bias also moved to GPU
                @test model_gpu.J_bias isa CuMatrix{Float32}

                out_cpu = smpl_lbs(model_cpu, betas, poses, trans)
                out_gpu = smpl_lbs(
                    model_gpu,
                    CuArray(betas),
                    CuArray(poses),
                    CuArray(trans),
                )

                # Output arrays on GPU
                @test out_gpu.vertices    isa CuMatrix{Float32}
                @test out_gpu.joints      isa CuMatrix{Float32}
                @test out_gpu.v_shaped    isa CuMatrix{Float32}
                @test out_gpu.v_posed     isa CuMatrix{Float32}
                # FK result and faces always on CPU
                @test out_gpu.J_transforms isa Array{Float32, 3}
                @test out_gpu.faces        === model_cpu.faces
                # Numerical agreement with CPU
                @test maximum(abs.(Array(out_gpu.vertices) .- out_cpu.vertices)) < GPU_ATOL
                @test maximum(abs.(Array(out_gpu.joints)   .- out_cpu.joints))   < GPU_ATOL
            end
        end

    end  # @testset "GPU (CUDA)"

end  # cuda_available
