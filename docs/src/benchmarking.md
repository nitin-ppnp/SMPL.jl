# Benchmarking

SMPL.jl ships a developer-only benchmarking suite in `bench/` that measures and
compares four execution backends (CPU Julia, GPU CUDA, static Julia, static compiled
exe) and breaks down the time spent in each of the 8 LBS pipeline steps.

## Setup

The benchmarks live in a separate isolated project so they don't pollute the
main package environment.  On first run, instantiate it once:

```bash
julia --project=bench/ -e 'using Pkg; Pkg.instantiate()'
```

This pins `SMPL` as a local path dependency.  No additional packages are
required — timing uses a built-in `@elapsed`-based loop (BenchmarkTools has
known `$`-interpolation issues with Julia 1.12's parser).

## Running

```bash
# CPU only (default)
julia --project=bench/ bench/run_benchmarks.jl

# CPU + GPU (requires CUDA.jl installed in the bench/ env)
SMPL_BENCH_GPU=true julia --project=bench/ bench/run_benchmarks.jl

# CPU + static Julia (GC-free pipeline, no exe needed — auto-detects .smplbin in repo root)
# Runs automatically whenever a .smplbin file is found for the selected model.
julia --project=bench/ bench/run_benchmarks.jl

# CPU + static executable (requires `julia compile.jl` to have been run first)
SMPL_BENCH_STATIC=true julia --project=bench/ bench/run_benchmarks.jl

# All backends, custom model
SMPL_BENCH_GPU=true SMPL_BENCH_STATIC=true SMPL_BENCH_MODEL=smplx_female \
  julia --project=bench/ bench/run_benchmarks.jl
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SMPL_BENCH_MODEL` | `smpl_female` | Model to load. Options: `smpl_{female,male,neutral}`, `smplx_{female,male,neutral}`, `supr_{female,male,neutral}` |
| `SMPL_BENCH_GPU` | `auto` | `true` forces GPU; `false` disables; `auto` runs GPU if CUDA is available |
| `SMPL_BENCH_STATIC` | `auto` | `true` forces static-exe; `false` disables; `auto` runs if `build/bin/smpl[.exe]` exists |
| `SMPL_BENCH_STATIC_MODEL` | *(auto)* | Path to `.smplbin` file. Auto-detected from model name if not set |
| `SMPL_BENCH_SECONDS` | `5` | Seconds per full-pipeline benchmark |
| `SMPL_BENCH_COMP_SECONDS` | `2` | Seconds per per-component benchmark |
| `SMPL_BENCH_STATIC_N` | `20` | Number of static-exe subprocess repetitions |

## Example Output

```
==================================================================
                    SMPL.jl Benchmark Report
==================================================================
  Model: smpl_female     │  β-dim: 10   │  θ-dim: 72   │  N_v: 6890
  Julia: 1.11.0          │  Threads: 1
==================================================================

[ Full Pipeline Comparison ]
┌────────────────────────┬────────────┬────────────┐
│ Backend                │ Min        │ Median     │
├────────────────────────┼────────────┼────────────┤
│ CPU (Julia)            │  12.34 ms  │  13.21 ms  │
│ GPU (CUDA)             │   1.23 ms  │   1.31 ms  │
│ Static (Julia)         │  18.50 ms  │  18.61 ms  │
│ Static (exe)           │ 163.00 ms  │ 165.00 ms  │
└────────────────────────┴────────────┴────────────┘
  GPU speedup vs CPU: 10.1×
  Static Julia vs CPU: 0.71×  ( 18.61 ms vs  13.21 ms)
  * Static Julia: GC-free pipeline, no BLAS (manual matrix loops)
  Static exe vs CPU: 0.08×  (165.00 ms vs  13.21 ms)
  * Static exe time includes subprocess launch + model load (cold start per call)

[ Per-Component Breakdown (CPU) ]
┌────────────────────────────────────────┬────────────┬───────┐
│ Step                                   │ Median     │ %     │
├────────────────────────────────────────┼────────────┼───────┤
│ (1) Shape blend shapes   v̄ + S·β      │   0.51 ms  │  3.9% │
│ (2) Joint positions      R_J · v_s    │   0.12 ms  │  0.9% │
│ (3) Per-joint rodrigues  R_k=exp(θ_k) │   0.23 ms  │  1.7% │
│ (4) Pose features        ψ=vec(Rᵀ−I)  │   0.08 ms  │  0.6% │
│ (5) Pose blend shapes    v_s + P·ψ    │   5.41 ms  │ 41.0% │
│ (6) Forward kinematics   FK(R, J)     │   0.45 ms  │  3.4% │
│ (7) Blend transforms     Σ w·G        │   3.12 ms  │ 23.6% │
│ (8) Linear blend skin    T·[v_p;1]    │   3.25 ms  │ 24.6% │
└────────────────────────────────────────┴────────────┴───────┘
  Component sum:  13.17 ms  │  Full pipeline:  13.21 ms  │  Overhead: 0.3%
```

## Static Benchmarks Prerequisite

Both the Static Julia and Static exe benchmarks need a `.smplbin` model file.
Convert once with:

```bash
julia scripts/convert_model.jl SMPL_FEMALE.npz SMPL_FEMALE.smplbin
```

The file is auto-detected in the repo root by model name
(e.g. `smpl_female` → `SMPL_FEMALE.smplbin`). Set
`SMPL_BENCH_STATIC_MODEL=/path/to/model.smplbin` to override.

The Static exe benchmark additionally requires a compiled executable:

```bash
julia compile.jl
```

The **Static Julia** benchmark (`Static (Julia)` row) runs the GC-free
`_static_smpl_lbs` pipeline directly in Julia — no compiled exe needed.
It benchmarks the algorithmic cost of manual matrix loops without BLAS or GC,
isolating that cost from subprocess launch and model-load overhead.

## Programmatic Use

The `BenchmarkSuite` module can be used directly for custom profiling:

```julia
include("bench/BenchmarkSuite.jl")
using .BenchmarkSuite, SMPL

model = create_smpl_female()
β     = zeros(Float32, 10)
θ     = zeros(Float32, 72)

# Full pipeline — returns a BenchResult(min_ns, median_ns, n_samples)
result = benchmark_pipeline(model, β, θ, zeros(Float32, 3); seconds=5)
@printf("median: %.2f ms\n", result.median_ns / 1e6)

# Per-component — returns a NamedTuple of BenchResult (BodyModel only)
comp = benchmark_components(model, β, θ; seconds=2)
@printf("Pose blend shapes: %.2f ms\n", comp.pose_blend.median_ns / 1e6)

# Static Julia — GC-free pipeline, no exe required (needs .smplbin)
sj = benchmark_static_julia("SMPL_FEMALE.smplbin", 10, 24; seconds=5)
@printf("Static Julia median: %.2f ms\n", sj.median_ns / 1e6)
```
