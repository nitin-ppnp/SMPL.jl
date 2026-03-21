# types.jl — Core data structures for the SMPL family of human body models.
#
# Struct types defined here:
#   BodyModel      — SMPL and SMPLX models (rotation-matrix pose features)
#   SUPRModel      — SUPR model (quaternion pose features, generalized joint regressor)
#   SMPLOutput     — type-stable return value of smpl_lbs (replaces Dict)
#   MotionSequence — loaded motion clip (pose/shape/trans per frame)
#
# All structs are parameterized over the array type A2, which can be:
#   Matrix{Float32}      — standard CPU computation
#   CuMatrix{Float32}    — NVIDIA GPU via CUDA.jl (after Adapt.adapt)
#   MtlMatrix{Float32}   — Apple Metal GPU via Metal.jl
#   MallocMatrix{Float32}— JuliaC trimmer path (see static/static_io.jl)
# The same smpl_lbs function handles all of these without any code changes,
# because all O(N_v) operations use standard AbstractArray broadcasting.
#
# `parents` and `faces` are always plain CPU Vector/Matrix; they are not
# parameterized because the kinematic chain is sequential (cannot run on GPU)
# and face indices are only consumed by the renderer.


# ---------------------------------------------------------------------------
# BodyModel — for SMPL and SMPLX
# ---------------------------------------------------------------------------

"""
    BodyModel{T<:AbstractFloat, A2<:AbstractMatrix{T}}

Unified parameterized model struct for SMPL (24 joints, 6890 vertices) and
SMPLX (55 joints, 10475 vertices). Both share identical LBS mechanics; they
differ only in the sizes stored in the arrays.

Type parameter:
  T  — element type, always Float32 in practice
  A2 — matrix backend; controls which device computation runs on.
       Construct a CPU model with `create_smpl*` / `create_smplx*`.
       Move to GPU with `Adapt.adapt(CuArray, model)` (requires CUDA.jl loaded).

Field layout follows SMPL paper notation (Loper et al., 2015, eq. 1–8):

  v̄  = v_template   (N_v × 3)         — mean body shape in rest pose
  S   = shapedirs    (N_v*3 × N_b)     — shape PCA basis (flattened vertices)
  P   = posedirs     ((N_j-1)*9 × N_v*3) — pose corrective basis
  R_J = J_regressor  (N_j × N_v)       — joint position regressor
  W   = lbs_weights  (N_v × N_j)       — per-vertex blend weights
      parents        (N_j,)  Int32, 1-indexed; parents[1]=1 (root→itself)
      faces          (N_f × 3) UInt32, 1-indexed triangle indices

All matrix fields share the same concrete array type A2, so the struct is
fully type-stable: no abstract-type fields, no dynamic dispatch in hot paths.
"""
struct BodyModel{T<:AbstractFloat, A2<:AbstractMatrix{T}}
    v_template  :: A2            # (N_v, 3)       — mean shape T̄
    shapedirs   :: A2            # (N_v*3, N_b)   — shape blend shape matrix S
    posedirs    :: A2            # ((N_j-1)*9, N_v*3) — pose blend shape matrix P
    J_regressor :: A2            # (N_j, N_v)     — joint regressor R_J
    lbs_weights :: A2            # (N_v, N_j)     — LBS weight matrix W
    parents     :: Vector{Int32} # (N_j,)  kinematic tree (always CPU, sequential FK)
    faces       :: Matrix{UInt32}# (N_f, 3) triangle mesh (always CPU, renderer only)
end


# ---------------------------------------------------------------------------
# SUPRModel — for SUPR
# ---------------------------------------------------------------------------

"""
    SUPRModel{T<:AbstractFloat, A2<:AbstractMatrix{T}}

Model struct for SUPR (75 joints, 10475 vertices), which differs from SMPL/SMPLX
in two fundamental ways:

  1. Pose features: quaternion-based (4D per joint → N_j*4 total) instead of
     the rotation-matrix-based (9D per joint, (N_j-1)*9 total) used by SMPL.

  2. Joint regressor: a generalized affine map from all vertex coordinates
     (not just a per-vertex scalar weight), split here into:
       J_regressor  (3*N_j, N_v*3)  — body weights (reshaped from sparse NPZ)
       J_bias       (N_j, 3)        — additive bias extracted from last column

Same A2 type parameter convention as BodyModel. Move to GPU with Adapt.

Field layout (SUPR paper notation):
  v̄  = v_template    (N_v, 3)
  S   = shapedirs     (N_v*3, N_b)  — reshaped from (N_v, 3, N_b) at load time
  P   = posedirs      (N_j*4, N_v*3)— reshaped from (N_v, 3, N_j*4) at load time;
                                       maps quaternion features to vertex offsets
  K   = J_regressor   (3*N_j, N_v*3)— affine body part of joint regressor
  b   = J_bias        (N_j, 3)      — additive joint bias
  W   = lbs_weights   (N_v, N_j)
      parents         (N_j,) Int32, 1-indexed
      faces           (N_f, 3) UInt32, 1-indexed
"""
struct SUPRModel{T<:AbstractFloat, A2<:AbstractMatrix{T}}
    v_template  :: A2            # (N_v, 3)
    shapedirs   :: A2            # (N_v*3, N_b)
    posedirs    :: A2            # (N_j*4, N_v*3) — quaternion pose corrective basis
    J_regressor :: A2            # (3*N_j, N_v*3) — generalized joint weights
    J_bias      :: A2            # (N_j, 3)       — additive joint position bias
    lbs_weights :: A2            # (N_v, N_j)
    parents     :: Vector{Int32} # (N_j,)  always CPU
    faces       :: Matrix{UInt32}# (N_f, 3) always CPU
end


# ---------------------------------------------------------------------------
# SMPLOutput — type-stable return from smpl_lbs
# ---------------------------------------------------------------------------

"""
    SMPLOutput{T<:AbstractFloat, A2<:AbstractMatrix{T}}

Type-stable output of `smpl_lbs`. Replaces the previous `Dict{String, Array}`
return, which caused dynamic dispatch at every field access and prevented the
compiler from inferring downstream types.

All matrix fields share the same array type A2 as the input BodyModel/SUPRModel,
so GPU outputs stay on the GPU without implicit transfers.

Fields:
  vertices     (N_v, 3)  — final skinned vertex positions (world space)
  joints       (N_j, 3)  — final posed joint positions (world space)
  v_shaped     (N_v, 3)  — vertices after shape blend shapes only
  v_posed      (N_v, 3)  — vertices after shape + pose blend shapes
  J_transforms (4, 4, N_j) — global 4×4 joint transforms G_k (always CPU Array)
  faces        Matrix{UInt32} — reference to model.faces (no copy)
"""
struct SMPLOutput{T<:AbstractFloat, A2<:AbstractMatrix{T}}
    vertices     :: A2               # (N_v, 3)  final posed + skinned vertices
    joints       :: A2               # (N_j, 3)  global joint positions
    v_shaped     :: A2               # (N_v, 3)  after shape blend shapes
    v_posed      :: A2               # (N_v, 3)  after pose blend shapes
    J_transforms :: Array{T, 3}      # (4, 4, N_j)  full 4×4 transforms (CPU)
    faces        :: Matrix{UInt32}   # reference into model.faces
end


# ---------------------------------------------------------------------------
# MotionSequence — loaded motion clip
# ---------------------------------------------------------------------------

"""
    MotionSequence{T<:AbstractFloat}

A loaded motion sequence from a `.smpl` (smplcodec) or AMASS `.npz` file.
`poses` is always stored as flat `(N_frames, pose_dim)` Float32, where
`pose_dim = N_joints * 3` (axis-angle, one 3-vector per joint).

`model_type` and `gender` indicate which model variant to use for forward
kinematics — pass the matching `BodyModel` / `SUPRModel` to `bake_motion`.

`up` indicates which world axis is "up" in the source coordinate system:
  - `:y` — Y-up (SMPL/AMASS default)
  - `:z` — Z-up (smplcodec v2 / some motion capture systems)

The visualizer uses `up` to orient the scene correctly.

Fields:
  poses      (N_frames, pose_dim)  — flat axis-angle pose per frame
  betas      (N_b,)                — shape parameters (constant across frames)
  trans      (N_frames, 3)         — root translation per frame
  fps        Float32               — capture/playback frame rate
  model_type Symbol                — :smpl | :smplx | :supr
  gender     Symbol                — :male | :female | :neutral
  up         Symbol                — :y | :z  (which world axis is "up")
"""
struct MotionSequence{T<:AbstractFloat}
    poses      :: Matrix{T}   # (N_frames, pose_dim)  flat axis-angle
    betas      :: Vector{T}   # (N_b,)
    trans      :: Matrix{T}   # (N_frames, 3)
    fps        :: Float32
    model_type :: Symbol      # :smpl | :smplx | :supr
    gender     :: Symbol      # :male | :female | :neutral
    up         :: Symbol      # :y | :z

    function MotionSequence{T}(poses, betas, trans, fps, model_type, gender,
                               up=:y) where T<:AbstractFloat
        new{T}(poses, betas, trans, fps, model_type, gender, up)
    end
end

# Outer convenience constructor — infers T from poses
function MotionSequence(poses::Matrix{T}, betas, trans, fps, model_type, gender,
                        up=:y) where T<:AbstractFloat
    MotionSequence{T}(poses, betas, trans, fps, model_type, gender, up)
end
