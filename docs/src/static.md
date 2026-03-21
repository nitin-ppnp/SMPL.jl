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

This produces `build/smpl.exe` (or `build/smpl` on Linux/macOS). The build uses JuliaC's `unsafe` trim mode, which strips the GC and IO dispatch table — the resulting binary has no Julia runtime dependency.

## Step 3 — Run

```bash
./build/smpl SMPL_MALE.smplbin
```

The executable runs `smpl_lbs` on a default zero pose and prints vertex and joint positions to stdout.

## Technical Notes

### Why a Separate Loader?

The trimmer strips Julia's IO dispatch table, so `NPZ.npzread`, `open(file)`, and all standard Julia IO is unavailable. `static/static_io.jl` works around this with direct C standard library calls:

```julia
fp  = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), path, "rb")
# ... read arrays with ccall(:fread, ...)
ccall(:fclose, Cint, (Ptr{Cvoid},), fp)
```

### Array Types

The static path instantiates `BodyModel{Float32, MallocMatrix{Float32}}` — `MallocMatrix` from StaticTools.jl allocates via `malloc` instead of the GC, making it available in trimmer mode. The same `smpl_lbs` function is reused because it only requires `AbstractMatrix{T}`.

### Extension Safety

`Adapt.jl` and `Makie.jl` must **not** appear in `static_project/Project.toml`. If they did, the corresponding extensions (`AdaptExt`, `MakieExt`) would load and attempt to `using Adapt` / `using Makie`, which triggers dynamic dispatch that the trimmer cannot handle.

`StaticArrays.jl` is safe — it is allocation-free and all `@inline` functions compile away completely.
```
