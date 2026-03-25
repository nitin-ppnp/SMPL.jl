# run_benchmarks.jl — SMPL.jl benchmark runner.
#
# Measures and compares performance of all enabled backends, then prints a
# formatted ASCII report to stdout.
#
# Usage:
#   julia --project=bench/ bench/run_benchmarks.jl
#
# Environment variables:
#   SMPL_BENCH_MODEL        model to load (default: "smpl_female")
#                           options: smpl_{female,male,neutral}
#                                    smplx_{female,male,neutral}
#                                    supr_{female,male,neutral}
#   SMPL_BENCH_GPU          enable GPU benchmark: "true" | "false" | "auto" (default: "auto")
#                           "auto" runs GPU if CUDA is available and functional
#   SMPL_BENCH_STATIC       enable static-exe benchmark: "true" | "false" | "auto" (default: "auto")
#                           "auto" runs if build/bin/smpl[.exe] exists
#   SMPL_BENCH_STATIC_MODEL path to .smplbin file (default: auto-detected from model name)
#   SMPL_BENCH_SECONDS      seconds per full-pipeline benchmark (default: 5)
#   SMPL_BENCH_COMP_SECONDS seconds per per-component benchmark (default: 2)
#   SMPL_BENCH_STATIC_N     number of static-exe timing repetitions (default: 20)

include(joinpath(@__DIR__, "BenchmarkSuite.jl"))
using .BenchmarkSuite
using SMPL
using Printf
using CUDA
using Adapt


# ===========================================================================
# Configuration
# ===========================================================================

const BENCH_MODEL        = get(ENV, "SMPL_BENCH_MODEL",        "smpl_female")
const BENCH_GPU          = get(ENV, "SMPL_BENCH_GPU",          "auto")
const BENCH_STATIC       = get(ENV, "SMPL_BENCH_STATIC",       "auto")
const BENCH_STATIC_MODEL = get(ENV, "SMPL_BENCH_STATIC_MODEL", "")
const BENCH_SECS         = parse(Int, get(ENV, "SMPL_BENCH_SECONDS",      "5"))
const BENCH_COMP_SECS    = parse(Int, get(ENV, "SMPL_BENCH_COMP_SECONDS", "2"))
const BENCH_STATIC_N     = parse(Int, get(ENV, "SMPL_BENCH_STATIC_N",     "20"))

const REPO_ROOT = dirname(@__DIR__)

const MODEL_LOADERS = Dict(
    "smpl_female"   => create_smpl_female,
    "smpl_male"     => create_smpl_male,
    "smpl_neutral"  => create_smpl_neutral,
    "smplx_female"  => create_smplx_female,
    "smplx_male"    => create_smplx_male,
    "smplx_neutral" => create_smplx_neutral,
    "supr_female"   => create_supr_female,
    "supr_male"     => create_supr_male,
    "supr_neutral"  => create_supr_neutral,
)

# Mapping from model name to expected .smplbin filename in repo root
const SMPLBIN_NAMES = Dict(
    "smpl_female"   => "SMPL_FEMALE.smplbin",
    "smpl_male"     => "SMPL_MALE.smplbin",
    "smpl_neutral"  => "SMPL_NEUTRAL.smplbin",
    "smplx_female"  => "SMPLX_FEMALE.smplbin",
    "smplx_male"    => "SMPLX_MALE.smplbin",
    "smplx_neutral" => "SMPLX_NEUTRAL.smplbin",
    "supr_female"   => "SUPR_FEMALE.smplbin",
    "supr_male"     => "SUPR_MALE.smplbin",
    "supr_neutral"  => "SUPR_NEUTRAL.smplbin",
)


# ===========================================================================
# Formatting helpers
# ===========================================================================

# Convert nanoseconds to a human-readable string with consistent width.
function _fmt_time(ns::Real) :: String
    ns < 1e3  && return @sprintf("%7.2f ns", ns)
    ns < 1e6  && return @sprintf("%7.2f μs", ns / 1e3)
    ns < 1e9  && return @sprintf("%7.2f ms", ns / 1e6)
    return          @sprintf("%7.2f  s", ns / 1e9)
end

# Print a horizontal rule of width w.
_rule(w) = println("=" ^ w)

# Left-pad a string s to width w.
_lpad(s::AbstractString, w::Int) = lpad(s, w)
_rpad(s::AbstractString, w::Int) = rpad(s, w)

# Print a table row: │ col1 │ col2 │ ... │
function _trow(cols::Vector{Pair{String,Int}})
    print("│")
    for (val, w) in cols
        print(" ", rpad(val, w), " │")
    end
    println()
end

# Print a separator row: ├─...─┼─...─┤
function _tsep(widths::Vector{Int}; top=false, bot=false)
    lc = top ? "┌" : (bot ? "└" : "├")
    rc = top ? "┐" : (bot ? "┘" : "┤")
    mc = top ? "┬" : (bot ? "┴" : "┼")
    print(lc)
    for (i, w) in enumerate(widths)
        print("─" ^ (w + 2))
        print(i < length(widths) ? mc : rc)
    end
    println()
end


# ===========================================================================
# Main
# ===========================================================================

function main()
    # ---- Validate model choice ----
    haskey(MODEL_LOADERS, BENCH_MODEL) || error(
        "Unknown SMPL_BENCH_MODEL=$BENCH_MODEL. " *
        "Options: $(join(sort(collect(keys(MODEL_LOADERS))), ", "))"
    )

    # ---- Load model ----
    println("\nLoading model: $BENCH_MODEL ...")
    model = MODEL_LOADERS[BENCH_MODEL]()

    N_v = size(model.v_template, 1)
    N_j = model isa BodyModel ? size(model.J_regressor, 1) : size(model.lbs_weights, 2)
    N_b = size(model.shapedirs, 2)
    ET  = Float32

    β     = zeros(ET, N_b)
    θ     = zeros(ET, N_j * 3)
    trans = zeros(ET, 3)

    # ---- Detect available backends ----
    run_gpu    = _should_run_gpu()
    run_static = _should_run_static()

    # ---- Run benchmarks ----
    println("Running CPU benchmark ($(BENCH_SECS)s) ...")
    cpu_trial = benchmark_pipeline(model, β, θ, trans; seconds=BENCH_SECS)

    gpu_trial = nothing
    if run_gpu
        println("Running GPU benchmark ($(BENCH_SECS)s) ...")
        try
            gpu_trial = _run_gpu_benchmark(model, β, θ, trans)
        catch e
            @warn "GPU benchmark failed: $e"
        end
    end

    static_result = nothing
    if run_static
        smplbin_path = _find_smplbin()
        if smplbin_path !== nothing
            exe_path = _static_exe_path()
            println("Running static-exe benchmark ($BENCH_STATIC_N reps) ...")
            try
                static_result = benchmark_static(exe_path, smplbin_path, β, θ, trans;
                                                  n=BENCH_STATIC_N)
            catch e
                @warn "Static benchmark failed: $e"
            end
        end
    end

    # Static Julia benchmark (GC-free pipeline, no exe required — only needs .smplbin)
    static_julia_trial = nothing
    if model isa BodyModel
        smplbin_path_julia = _find_smplbin()
        if smplbin_path_julia !== nothing
            println("Running static Julia benchmark ($(BENCH_SECS)s) ...")
            try
                static_julia_trial = benchmark_static_julia(smplbin_path_julia, N_b, N_j;
                                                             seconds=BENCH_SECS)
            catch e
                @warn "Static Julia benchmark failed: $e"
            end
        end
    end

    comp_results = nothing
    if model isa BodyModel
        println("Running per-component breakdown ($(BENCH_COMP_SECS)s each) ...")
        comp_results = benchmark_components(model, β, θ; seconds=BENCH_COMP_SECS)
    else
        println("Note: per-component breakdown skipped for SUPRModel (BodyModel only).")
    end

    # ---- Print report ----
    _print_report(cpu_trial, gpu_trial, static_result, static_julia_trial, comp_results,
                  N_v, N_j, N_b)
end


# ===========================================================================
# Backend detection helpers
# ===========================================================================

function _should_run_gpu()
    BENCH_GPU == "true"  && return true
    BENCH_GPU == "false" && return false
    # "auto": run if CUDA is functional
    return CUDA.functional()
end

function _should_run_static()
    BENCH_STATIC == "true"  && return true
    BENCH_STATIC == "false" && return false
    # "auto": check if exe exists
    return isfile(_static_exe_path())
end

function _static_exe_path()
    exe = Sys.iswindows() ? "smpl.exe" : "smpl"
    joinpath(REPO_ROOT, "build", "bin", exe)
end

function _find_smplbin()
    # User-specified path takes priority
    !isempty(BENCH_STATIC_MODEL) && return BENCH_STATIC_MODEL

    # Auto-detect from model name
    fname = get(SMPLBIN_NAMES, BENCH_MODEL, "")
    isempty(fname) && return nothing
    path  = joinpath(REPO_ROOT, fname)
    isfile(path) ? path : nothing
end

function _run_gpu_benchmark(model, β, θ, trans)
    gpu_model = Adapt.adapt(CuArray, model)
    return benchmark_pipeline_gpu(gpu_model, CuArray(β), CuArray(θ), CuArray(trans),
                                   CUDA.synchronize; seconds=BENCH_SECS)
end


# ===========================================================================
# Report printing
# ===========================================================================

function _print_report(cpu_trial, gpu_trial, static_result, static_julia_trial, comp_results,
                       N_v, N_j, N_b)
    julia_ver = string(VERSION)
    nthreads  = Threads.nthreads()
    W = 66  # total report width

    println()
    _rule(W)
    println(lpad("SMPL.jl Benchmark Report", (W + 24) ÷ 2))
    _rule(W)
    @printf("  Model: %-14s │  β-dim: %-4d │  θ-dim: %-4d │  N_v: %d\n",
            BENCH_MODEL, N_b, N_j * 3, N_v)
    @printf("  Julia: %-10s     │  Threads: %d\n", julia_ver, nthreads)
    _rule(W)

    # ---- Full pipeline comparison ----
    println("\n[ Full Pipeline Comparison ]")
    col_w = [22, 10, 10]   # Backend | Min | Median
    _tsep(col_w; top=true)
    _trow(["Backend" => col_w[1], "Min" => col_w[2], "Median" => col_w[3]])
    _tsep(col_w)

    _trow([
        "CPU (Julia)"               => col_w[1],
        _fmt_time(cpu_trial.min_ns) => col_w[2],
        _fmt_time(cpu_trial.median_ns) => col_w[3],
    ])

    if gpu_trial !== nothing
        _trow([
            "GPU (CUDA)"                   => col_w[1],
            _fmt_time(gpu_trial.min_ns)    => col_w[2],
            _fmt_time(gpu_trial.median_ns) => col_w[3],
        ])
    end

    if static_julia_trial !== nothing
        _trow([
            "Static (Julia)"                          => col_w[1],
            _fmt_time(static_julia_trial.min_ns)      => col_w[2],
            _fmt_time(static_julia_trial.median_ns)   => col_w[3],
        ])
    end

    if static_result !== nothing
        _trow([
            "Static (exe)"                          => col_w[1],
            _fmt_time(static_result.min_s * 1e9)    => col_w[2],
            _fmt_time(static_result.median_s * 1e9) => col_w[3],
        ])
    end

    _tsep(col_w; bot=true)

    # Speedup summary lines
    if gpu_trial !== nothing
        speedup = cpu_trial.median_ns / gpu_trial.median_ns
        @printf("  GPU speedup vs CPU: %.1f×\n", speedup)
    end
    if static_julia_trial !== nothing
        ratio = cpu_trial.median_ns / static_julia_trial.median_ns
        @printf("  Static Julia vs CPU: %.2f×  (%s vs %s)\n",
                ratio,
                _fmt_time(static_julia_trial.median_ns),
                _fmt_time(cpu_trial.median_ns))
        println("  * Static Julia: GC-free pipeline, no BLAS (manual matrix loops)")
    end
    if static_result !== nothing
        ratio = cpu_trial.median_ns / (static_result.median_s * 1e9)
        @printf("  Static exe vs CPU: %.2f×  (%s vs %s)\n",
                ratio,
                _fmt_time(static_result.median_s * 1e9),
                _fmt_time(cpu_trial.median_ns))
        println("  * Static exe time includes subprocess launch + model load (cold start per call)")
    end

    # ---- Per-component breakdown ----
    if comp_results !== nothing
        println("\n[ Per-Component Breakdown (CPU) ]")

        step_labels = [
            "(1) Shape blend shapes   v̄ + S·β",
            "(2) Joint positions      R_J · v_s",
            "(3) Per-joint rodrigues  R_k = exp(θ_k)",
            "(4) Pose features        ψ = vec(Rᵀ−I)",
            "(5) Pose blend shapes    v_s + P·ψ",
            "(6) Forward kinematics   FK(R, J)",
            "(7) Blend transforms     Σ w·G",
            "(8) Linear blend skin    T·[v_p;1]",
        ]
        step_results = [
            comp_results.shape_blend,
            comp_results.joint_pos,
            comp_results.rodrigues,
            comp_results.pose_features,
            comp_results.pose_blend,
            comp_results.fwd_kin,
            comp_results.blend_xforms,
            comp_results.skinning,
        ]

        # Compute per-step medians (ns)
        step_ns = [r.median_ns for r in step_results]
        total_ns = sum(step_ns)

        cw = [38, 10, 7]   # Step | Median | %
        _tsep(cw; top=true)
        _trow(["Step" => cw[1], "Median" => cw[2], "%" => cw[3]])
        _tsep(cw)

        for (label, ns) in zip(step_labels, step_ns)
            pct = 100.0 * ns / total_ns
            _trow([
                label                      => cw[1],
                _fmt_time(ns)              => cw[2],
                @sprintf("%5.1f%%", pct)   => cw[3],
            ])
        end

        _tsep(cw; bot=true)
        @printf("  Component sum: %s  │  Full pipeline: %s  │  Overhead: %.1f%%\n",
                _fmt_time(total_ns),
                _fmt_time(cpu_trial.median_ns),
                100.0 * (cpu_trial.median_ns - total_ns) / cpu_trial.median_ns)
    end

    println()
    _rule(W)
    println()
end


# ===========================================================================
# Entry point
# ===========================================================================

main()
