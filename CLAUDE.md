# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Build docs locally
julia --project=docs/ docs/make.jl
python -m http.server 8000 -d docs/build

# Convert NPZ model to static binary
julia scripts/convert_model.jl input.npz output.smplbin

# Build static executable
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

## Rules

- `.claude/rules/architecture.md` — file layout, struct fields, GPU/CPU boundary, format detection
- `.claude/rules/code_style.md` — parameterized structs, docstrings, GPU compatibility, NPZ reshape rule
- `.claude/rules/commands.md` — package management, test, docs, static build commands

## Session rules

See `.claude/rules/meta.md` for end-of-session requirements (`/review`, `/update-docs`, architecture update).
