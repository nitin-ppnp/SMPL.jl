# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Skip the static IO test (if SMPL models not downloaded)
SMPL_TEST_STATIC=false julia --project=. -e 'using Pkg; Pkg.test()'

# Run an individual test file
julia --project=. test/test_static_io.jl

# Run benchmarks
julia --project=bench/ bench/run_benchmarks.jl

# Build docs locally
julia --project=docs/ docs/make.jl
python -m http.server 8000 -d docs/build

# Convert NPZ model to static binary
julia scripts/convert_model.jl input.npz output.smplbin

# Build static executable (outputs build/bin/smpl[.exe])
julia compile.jl
```

## Architecture

See `.claude/rules/architecture.md` for detailed notes. Key structure:

- `src/` — core Julia module (`types.jl`, `math.jl`, `io.jl`, `model.jl`, `motion_io.jl`)
- `ext/MakieExt.jl` — visualization (weak dep, loaded when GLMakie/CairoMakie/WGLMakie is imported)
- `ext/AdaptExt.jl` — GPU transfer via `Adapt.adapt(CuArray, model)`
- `static/` — JuliaC trimmer-safe static binary loader
- `docs/src/` — Documenter.jl pages; keep in sync with any API changes

## LBS Pipeline

8-step pipeline in `src/model.jl`: shape blend → joint positions → per-joint rotations → pose features → pose blend → forward kinematics (CPU-only) → blend transforms → skinning. SUPR differs at steps 2–4 (affine J_regressor + quaternion features).

## Model Dimensions Quick Reference

| Model | Joints | Vertices | Pose dim | Beta dim |
|-------|--------|----------|----------|----------|
| SMPL  | 24     | 6890     | 72       | 10       |
| SMPLX | 55     | 10475    | 165      | 10       |
| SUPR  | 75     | 10475    | 225      | 10       |

SUPR note: AMASS stores SUPR with 228 = 76×3 pose elements (extra joint appended). `smpl_lbs` accepts both 225 and 228; extra elements beyond 225 are silently ignored.

`smpl_lbs` returns `SMPLOutput` (a struct, not a Dict). Access fields as `out.vertices`, `out.joints`. **The README.md shows outdated Dict-style access — do not use it as a reference.**

## Critical NPZ Reshape Rule

When collapsing a 3D `(N,J,D)` NPZ array to `(N, J*D)`, always permute first:
```julia
# CORRECT
reshape(permutedims(Float32.(A), (1, 3, 2)), N, :)
# WRONG — Julia column-major interleaves frames, not joints
reshape(Float32.(A), N, :)
```

## Rules

- `.claude/rules/architecture.md` — file layout, struct fields, GPU/CPU boundary, format detection
- `.claude/rules/code_style.md` — parameterized structs, docstrings, GPU compatibility, NPZ reshape rule
- `.claude/rules/commands.md` — package management, test, docs, static build commands

## Session rules

See `.claude/rules/meta.md` for end-of-session requirements (`/review`, `/update-docs`, architecture update).
