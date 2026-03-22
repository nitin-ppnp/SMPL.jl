# SMPL.jl Architecture Reference

## File Structure

```
src/
  SMPL.jl        # module root — includes/exports only; declares viz stubs
  types.jl       # BodyModel{T,A2}, SUPRModel{T,A2}, SMPLOutput{T,A2}, MotionSequence{T}
  math.jl        # rodrigues, quat_feat, forward_kinematics
  io.jl          # create_smpl/smplx/supr + DataDeps __init__ + _read_credentials
  model.jl       # smpl_lbs (BodyModel + SUPRModel), pivot_fk, _lbs_skin
  motion_io.jl   # load_motion — .smpl (smplcodec) + AMASS NPZ

ext/
  AdaptExt.jl    # Adapt.jl weak dep — GPU transfer via Adapt.adapt(CuArray, model)
  MakieExt.jl    # Makie.jl weak dep — bake_motion, viz_motion, viz_motions,
                 #                     record_motion, render_frame

static/
  static_io.jl   # JuliaC trimmer-safe loader — ccall(:fread,...), MallocMatrix

staticSMPL.jl    # @main entrypoint for compiled executable — GC-free static_smpl_lbs
compile.jl       # JuliaC build script — outputs build/bin/smpl[.exe] via bundle_products

scripts/
  convert_model.jl  # NPZ → .smplbin one-time converter; guarded with abspath(__FILE__) check

static_project/
  Project.toml   # Minimal deps for trimmer: LinearAlgebra, StaticArrays, StaticTools
  Manifest.toml  # Pinned manifest (regenerate with: julia --project=static_project -e 'using Pkg; Pkg.resolve()')

test/
  runtests.jl           # Outer @testset "SMPL.jl" wrapper; includes static tests
  test_static_io.jl     # Binary roundtrip test — no JuliaC needed; gated by SMPL_TEST_STATIC
  test_static_compile.jl # Full compile→run→verify test; gated by SMPL_TEST_COMPILE=true

docs/
  make.jl        # Documenter.jl build script
  Project.toml   # Documenter dep
  src/*.md       # Documentation pages (index, guide, api, visualization, gpu, static)

.github/workflows/docs.yml   # Build + deploy docs to gh-pages on push to master
credentials.toml.example     # Template — copy to credentials.toml (gitignored)
```

## Core Structs (src/types.jl)

- **`BodyModel{T, A2<:AbstractMatrix{T}}`** — SMPL and SMPLX. Type param `A2` determines backend.
  Fields: `v_template`, `shapedirs`, `posedirs`, `J_regressor`, `lbs_weights` (all `A2`);
  `parents::Vector{Int32}`, `faces::Matrix{UInt32}` (always CPU).

- **`SUPRModel{T, A2<:AbstractMatrix{T}}`** — SUPR. Adds `J_bias::A2 (N_j,3)`.
  `J_regressor` is `(3*N_j, N_v*3)` affine map; `posedirs` is `(N_j*4, N_v*3)` quaternion basis.

- **`SMPLOutput{T, A2}`** — return type of `smpl_lbs`. Fields: `vertices`, `joints`, `v_shaped`,
  `v_posed` (all `A2`); `J_transforms::Array{T,3}` (always CPU); `faces::Matrix{UInt32}`.

- **`MotionSequence{T}`** — motion clip. Fields: `poses (N,pose_dim)`, `betas (N_b,)`,
  `trans (N,3)`, `fps`, `model_type`, `gender`, `up` (`:y`|`:z` — which axis is "up").
  Has inner constructor with `up=:y` default; existing 6-arg call sites stay valid.

## LBS Pipeline (src/model.jl) — 8 steps

```
(1) v_s = v̄ + S·β              shape blend shapes
(2) J   = R_J · v_s             joint positions
(3) R_k = rodrigues(θ_k)        per-joint rotation matrices
(4) ψ   = vec(R_{2..K}ᵀ − I)   pose feature vector
(5) v_p = v_s + P·ψ             pose blend shapes
(6) FK(R, J, parents)           forward kinematics — ALWAYS runs on CPU
(7) T_i = Σ_k w_{ki}·A_k       per-vertex blend transform
(8) v_i = T_i·[v_p_i; 1]       linear blend skinning
```

SUPR differs: step 2 uses affine J_regressor + bias; steps 3-4 use `quat_feat` (4D quaternion features).

## GPU / CPU Boundary

- FK (step 6) runs on CPU — sequential, cannot parallelise across GPU threads.
- All other steps use `AbstractArray` dispatch and run on the device.
- `_lbs_skin(::Array, ...)` → BLAS per-vertex `mul!` (CPU); `_lbs_skin(::AbstractArray, ...)` → fused broadcast (GPU).

## load_motion Format Detection (`src/motion_io.jl`)

Auto-detected by key presence in the NPZ/SMPL file:

| Key present | Format | Up axis | Pose shape |
|---|---|---|---|
| `fullpose` | smplcodec v1 | `:y` | `(N,J,3)` → flat `(N,J*3)` |
| `bodyPose` | smplcodec v2 (SMPLX) | `:z` | body+head+hands → `(N,165)` |
| `poses` | AMASS npz | `:z` | already flat `(N,pose_dim)` |

All three formats are Z-up in world coordinates: the root joint pose rotates the SMPL Y-up T-pose
to Z-up world, so `smpl_lbs` output vertices have Z as height. Makie's LScene camera is also Z-up
by default, so no vertex rotation is needed. The `up` field can be set to `:y` by callers loading
data from Y-up sources.

## Visualization Extension (`ext/MakieExt.jl`)

Internal helpers (not exported):
- `_motion_rect(seq)` — computes a `Rect3f` bounding the full trajectory with body-width margins (~0.5 m XY, 0.1 m Z below floor, 0.2 m above head)
- `_ground_plane!(scene, rect)` — white floor quad at rect bounds + gray grid lines; stays within rect to avoid Makie auto-limit expansion
- `_origin_marker!(scene, rect)` — RGB XYZ arrows at world `(0,0,0)`, but **only drawn when `(0,0,0)` falls within `rect`**. Skips drawing when body is far from world origin to prevent Makie auto-limit expansion zooming the camera out
- `_setup_camera!(lscene, rect; camera_eye, camera_lookat, camera_upvector, camera_fov)` — positions `Camera3D` explicitly; **must be called after all geometry** (`poly!`, `lines!`, `mesh!`, `arrows3d!`) so Makie's `update_limits!` cannot override it afterward

Camera kwargs on all 4 public functions: `camera_eye::Union{Nothing,Vec3f}`, `camera_lookat::Union{Nothing,Vec3f}`, `camera_upvector::Vec3f`, `camera_fov::Float32`. Flat kwargs (not a struct) enable idiomatic splatting: `cam = (;camera_eye=..., camera_fov=35f0); viz_motion(model, seq; cam...)`.

**Makie limit-expansion pitfall**: every `poly!`, `lines!`, `mesh!`, `arrows3d!` call triggers `update_limits!(scene)`, which auto-frames the camera to encompass all geometry. Any geometry added outside the trajectory rect (e.g., grid lines with extra padding, markers at world origin) zooms the camera far out. Keep all scene geometry within `rect`.

## Static Compilation (`staticSMPL.jl` + `compile.jl`)

- `staticSMPL.jl` contains `static_smpl_lbs` — a fully GC-free reimplementation of the LBS pipeline using `MallocMatrix{Float32}` for all intermediates and `SMatrix{4,4}` for per-joint transforms. **Do not reuse `smpl_lbs` from `src/model.jl`** — it calls `zeros`, `ones`, `vcat`, `copy`, and BLAS, all of which are unavailable in `trim_mode="unsafe"`.
- `compile.jl` calls `bundle_products` which places the binary at **`build/bin/smpl[.exe]`** (not `build/smpl[.exe]`).
- `scripts/convert_model.jl` must write `parents` as **`Int32`** (dtype tag `3`) with the root sentinel zeroed before the `+1` 0→1-index offset. Writing `UInt32` causes silent bit-reinterpretation in `_fread_i32_vec`.
- `static_project/Manifest.toml` must be kept in sync with `Project.toml`. After adding a dep, run `julia --project=static_project -e 'using Pkg; Pkg.resolve()'`.
- Two test tiers: `test_static_io.jl` (no compiler, ~seconds) and `test_static_compile.jl` (`SMPL_TEST_COMPILE=true`, ~10 min).

## Key Conventions

- `parents` is 1-indexed `Int32`; root has `parents[1] = 1` (self-referencing).
- `faces` is 1-indexed `UInt32` (converted from 0-indexed NPZ at load time).
- SMPL/SMPLX NPZ root parent sentinel `0xffffffff` is forced to 0 before Int32 conversion.
- SUPR `J_regressor` is sparse `(3*N_j, N_v*3+1)` in NPZ — last column extracted as `J_bias`.
- **NPZ 3D reshape rule**: See `code_style.md` — use `reshape(permutedims(A,(1,3,2)), N, :)`, never plain `reshape(A, N, :)`.
