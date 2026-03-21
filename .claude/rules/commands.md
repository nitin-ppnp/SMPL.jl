# SMPL.jl Common Commands

## Package Management

```julia
# Add package (unregistered)
]add https://github.com/nitin-ppnp/SMPL.jl

# Run all tests (requires SMPL and SMPLX models downloaded; SUPR is optional)
]test
```

## Static Compilation

```bash
# Convert NPZ model to static binary format
julia scripts/convert_model.jl input.npz output.smplbin

# Build standalone static executable (outputs build/smpl.exe or build/smpl on Linux/macOS)
julia compile.jl
```

## Docs

```bash
# Build docs locally (from repo root)
julia --project=docs/ docs/make.jl

# Serve built docs at http://localhost:8000
python -m http.server 8000 -d docs/build
```

## Tests

Tests live in `test/runtests.jl`. They compare `out.vertices` and `out.joints`
(both `(N, 3)`) against reference `.npz` files with **1e-5 tolerance**.

- SMPL and SMPLX tests run against bundled reference outputs — no download needed.
- The SUPR test requires SUPR model files to be downloaded (prompts for credentials on first run,
  or reads from `credentials.toml`).
