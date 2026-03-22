# Static Compilation

SMPL.jl supports ahead-of-time compilation into a standalone executable via [JuliaC](https://github.com/JuliaCompilerInfrastructure/JuliaC.jl) (the Julia compiler infrastructure / "trimmer" path). This produces a self-contained binary with no Julia installation required at runtime.

## Overview

The static path uses a separate, minimal code entry point that avoids all runtime dependencies (NPZ.jl, DataDeps.jl, HTTP.jl, GLMakie, etc.) and loads body model data from a compact binary `.smplbin` format instead of `.npz` files.

Key components:
- `staticSMPL.jl` — `@main` entry point for the compiled executable
- `static/static_io.jl` — trimmer-safe model loader (uses `ccall(:fread,...)`)
- `compile.jl` — JuliaC build script
- `scripts/convert_model.jl` — converts `.npz` → `.smplbin`
- `static_project/Project.toml` — minimal dependencies for the static build

## Step 1 — Convert a Model to `.smplbin`

```bash
julia scripts/convert_model.jl SMPL_MALE.npz SMPL_MALE.smplbin
```

The `.smplbin` format is a compact binary with a fixed header and raw float arrays:

```
"SMPL" magic (4 bytes) + version UInt32 (4 bytes)
then for each array:
  ndims  (UInt8)
  shape  (UInt64 × ndims)
  dtype  (UInt8)
  raw data
```

Arrays stored in order: `v_template`, `shapedirs`, `posedirs`, `J_regressor`, `parents`, `lbs_weights`, `faces`.

## Step 2 — Build the Executable

```bash
julia compile.jl
```

This produces `build/bin/smpl.exe` (or `build/bin/smpl` on Linux/macOS), with bundled runtime libraries under `build/`. The build uses JuliaC's `unsafe` trim mode, which strips the GC and IO dispatch table — the resulting binary has no Julia installation required at runtime.

## Step 3 — Run

The executable has two modes:

**Interactive mode** (1 arg) — runs a zero-pose forward pass and prints statistics to stdout:

```bash
./build/bin/smpl SMPL_MALE.smplbin
```

**Binary I/O mode** (6 args) — reads inputs from binary files, writes outputs to binary files:

```bash
./build/bin/smpl SMPL_MALE.smplbin betas.bin poses.bin trans.bin verts_out.bin joints_out.bin
```

Input binary format: `UInt64` length-prefix + raw `Float32` bytes.
Output binary format: `UInt64` rows + `UInt64` cols + raw `Float32` bytes (column-major).

This binary I/O mode is used by `test/test_static_compile.jl` to verify outputs against the Python reference.

## Technical Notes

### Why a Separate Loader?

The trimmer strips Julia's IO dispatch table, so `NPZ.npzread`, `open(file)`, and all standard Julia IO is unavailable. `static/static_io.jl` works around this with direct C standard library calls:

```julia
fp  = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), path, "rb")
# ... read arrays with ccall(:fread, ...)
ccall(:fclose, Cint, (Ptr{Cvoid},), fp)
```

### LBS Pipeline in the Executable

The static executable uses a dedicated `static_smpl_lbs` function in `staticSMPL.jl` that is fully GC-free: no `zeros`, `ones`, `vcat`, `copy`, or BLAS calls. All intermediate matrices are `MallocMatrix{Float32}` (malloc-backed, no GC), and the forward kinematics loop uses manual stack-allocated `SMatrix{4,4,Float32}` operations from StaticArrays.jl.

`BodyModel{Float32, MallocMatrix{Float32}}` is loaded by `static/static_io.jl` via `ccall(:fread,...)` into malloc-backed arrays. The same struct layout is used as the normal CPU path (`BodyModel{Float32, Matrix{Float32}}`), so the type is identical — only the array backend differs.

### Array Types

The static path instantiates `BodyModel{Float32, MallocMatrix{Float32}}` — `MallocMatrix` from StaticTools.jl allocates via `malloc` instead of the GC, making it available in trimmer mode.

## Testing

Two test files cover the static path:

**`test/test_static_io.jl`** — runs without JuliaC. Calls the real `convert_npz_to_bin` converter, loads the result with `create_smpl`, verifies array shapes and values, then runs the forward pass (materialising `MallocMatrix → Matrix` first) against the Python reference outputs at 1e-5 tolerance. Enable with:

```bash
# Runs automatically as part of the full test suite:
julia --project=. -e 'using Pkg; Pkg.test()'

# Or disable with:
SMPL_TEST_STATIC=false julia --project=. -e 'using Pkg; Pkg.test()'
```

**`test/test_static_compile.jl`** — full end-to-end pipeline: convert → compile with JuliaC → run binary with binary I/O → compare outputs vs Python reference (1e-4 tolerance). Disabled by default due to ~10 min compile time:

```bash
SMPL_TEST_COMPILE=true julia --project=. test/test_static_compile.jl
```

### Extension Safety

`Adapt.jl` and `Makie.jl` must **not** appear in `static_project/Project.toml`. If they did, the corresponding extensions (`AdaptExt`, `MakieExt`) would load and attempt to `using Adapt` / `using Makie`, which triggers dynamic dispatch that the trimmer cannot handle.

`StaticArrays.jl` is safe — it is allocation-free and all `@inline` functions compile away completely.
