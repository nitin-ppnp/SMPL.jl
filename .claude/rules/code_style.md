# SMPL.jl Code Style Rules

These rules apply to all Julia source files in `src/`, `ext/`, and `static/`.

## Structs

- Use parameterized structs: `struct Foo{T, A2<:AbstractMatrix{T}}` — never abstract types as fields.
- `parents::Vector{Int32}` and `faces::Matrix{UInt32}` are always plain CPU arrays; do not parameterize them.
- New struct definitions go in `src/types.jl` only.

## Functions

- Every public function (exported or appearing in a paper) gets a docstring citing the equation, e.g. `R = I + sin(θ)K + (1-cos(θ))K²`.
- Label each LBS pipeline step with `# (1)`, `# (2)`, ..., `# (8)` matching the SMPL paper.
- Annotate array shapes as inline comments: `J = model.J_regressor * v_shaped  # (N_j, 3)`.
- No hardcoded sizes (`6890`, `24`, `10475`, `55`) in function bodies — derive from array dimensions.

## GPU Compatibility

- All hot-path functions take `AbstractMatrix{T}` / `AbstractArray{T}` — never concrete `Array`.
- CPU-only operations (sequential kinematic chain, IO) call `Array(x)` to materialise before the loop.
- Small fixed-size matrices (3×3 rotations) use `StaticArrays.SMatrix` — zero allocation, inlines into kernels.
- Dual dispatch pattern for device-specific kernels:
  ```julia
  f(x::Array, ...)         = ...  # CPU: scalar loop + BLAS mul!
  f(x::AbstractArray, ...) = ...  # GPU: fused broadcast
  ```

## No Dynamic Dispatch

- Return `SMPLOutput` struct, never `Dict`.
- All type parameters must be concrete at call sites.
- Run `@code_warntype smpl_lbs(model, β, θ)` after changes to confirm no yellow/red.

## GPU Adaptation

- Use `Adapt.jl` (weak dependency, loaded via `ext/AdaptExt.jl`).
- To move model to GPU: `gpu_model = Adapt.adapt(CuArray, model)`.
- `parents` and `faces` are excluded from adaptation (stay on CPU).

## Static Compilation

- `Adapt` must not appear in `static_project/Project.toml` (extension never loads → trimmer safe).
- Binary `.smplbin` loader in `static/static_io.jl` uses `ccall(:fread,...)` — no Julia IO dispatch.
- `StaticArrays` is safe for the trimmer (allocation-free, all `@inline`).

## NPZ Array Reshaping

When collapsing a 3D `(N,J,D)` NPZ array to a flat `(N, J*D)` matrix, always permute first:

```julia
# CORRECT — matches NumPy C-order reshape(N, -1)
reshape(permutedims(Float32.(A), (1, 3, 2)), N, :)

# WRONG — Julia column-major collapse interleaves frames, not joints
reshape(Float32.(A), N, :)
```

The permute `(1,3,2)` converts `(N,J,D)→(N,D,J)` so Julia's column-major flatten produces
`[j0_x, j0_y, j0_z, j1_x, ...]` per row, matching NumPy's default C-order `reshape(N,-1)`.

## Documentation

- Whenever a public API, function signature, docstring, or module changes, update `docs/src/` accordingly.
- `docs/src/api.md` must list all exported functions.
- All code examples in docs must be runnable and match the current API.
