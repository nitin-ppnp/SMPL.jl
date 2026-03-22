# motion_io.jl — Loading motion sequences from .smpl and AMASS .npz files.
#
# Public API:
#   load_motion(path)  -> MotionSequence{Float32}
#
# Supported formats (auto-detected by key presence in the NPZ):
#
#   smplcodec v1 (.smpl):
#     Keys: fullpose (N, N_j, 3),  transl (N, 3),  betas (N_b,)
#     Optional: gender (string), fps (scalar)
#
#   smplcodec v2 (.smpl, smplVersion=2):
#     Keys: bodyPose (N,22,3), headPose (N,3,3),
#           leftHandPose (N,15,3), rightHandPose (N,15,3),
#           bodyTranslation (N,3), frameRate (scalar), gender (Int32)
#     Note: no betas field — defaults to zeros(10)
#     Joint order: body(0-21) | head(22-24) | leftHand(25-39) | rightHand(40-54)
#
#   AMASS (.npz):
#     Keys: poses (N, pose_dim),  trans (N, 3),  betas (N_b,)
#     Optional: gender (string), mocap_framerate (scalar)
#
# In both cases the output `poses` field is always flat (N_frames, pose_dim),
# and `betas` is always (N_b,) — the first frame's betas for per-frame AMASS files.

using NPZ


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _parse_gender(s) -> Symbol

Normalise gender strings from various dataset conventions to :male / :female / :neutral.
Anything unrecognised maps to :neutral.
"""
function _parse_gender(s) :: Symbol
    g = lowercase(string(s))
    g == "male"    && return :male
    g == "female"  && return :female
    g == "m"       && return :male
    g == "f"       && return :female
    return :neutral
end

"""
    _infer_model_type(pose_dim) -> Symbol

Infer which body model a flat pose vector belongs to from its dimensionality:
  72  = SMPL  (24 joints × 3)
  165 = SMPLX (55 joints × 3)
  228 = SUPR  (76 joints × 3, including root)
Anything else is left as :smpl (best-effort).
"""
function _infer_model_type(pose_dim::Int) :: Symbol
    pose_dim == 72  && return :smpl
    pose_dim == 165 && return :smplx
    pose_dim == 228 && return :supr
    return :smpl
end


# ---------------------------------------------------------------------------
# load_motion
# ---------------------------------------------------------------------------

"""
    load_pivot_labels(path::String) -> Matrix{Float32}

Load per-frame pivot joint probability scores from a `.npy` file.

Returns a `(N_frames, N_tracked)` Float32 matrix where `N_tracked` is the
number of joints with scores (typically 23 for SMPLX body joints, corresponding
to model joint indices 1–23 in Julia's 1-indexed convention).

Values near 1.0 indicate high pivot probability; negative values indicate low
probability.

# Example
```julia
labels = load_pivot_labels("walk_stageii.npy")   # (N_frames, 23) Matrix{Float32}
```
"""
function load_pivot_labels(path::String) :: Matrix{Float32}
    return Float32.(NPZ.npzread(path))   # NPZ.npzread on .npy returns the array directly
end


"""
    load_motion(path::String; pivot_labels_path=nothing, up=nothing) -> MotionSequence{Float32}

Load a motion sequence from a `.smpl` (smplcodec) or AMASS `.npz` file.

Format is auto-detected by key presence:
  - `fullpose` key  → smplcodec v1: reshape `(N, N_j, 3)` to flat `(N, N_j*3)`
  - `bodyPose` key  → smplcodec v2 (SMPLX): concatenate body/head/hand parts → `(N, 165)`
  - `poses` key     → AMASS format: use as-is `(N, pose_dim)`

`betas` uses the file's betas if present; defaults to `zeros(10)` for smplcodec v2.
Per-frame beta arrays (AMASS shape `(N, N_b)`) use the first frame.

`fps` is read from `frameRate` (smplcodec v2), `fps` (v1), or `mocap_framerate` (AMASS);
defaults to 30.

`gender` is read from `gender` key if present; Int32 (smplcodec v2: 1=female, 2=male,
else neutral) or string (AMASS); defaults to `:neutral`.

`model_type` is inferred from `pose_dim` (72→:smpl, 165→:smplx, 228→:supr).

`up`: which world axis is "up" in the output of `smpl_lbs` for this file.
  Auto-detected when `nothing` (default):
    - smplcodec v1 (`fullpose` key) → `:y`  (root rotation not encoding Y→Z flip)
    - smplcodec v2 (`bodyPose` key) → `:z`  (root rotation encodes Y→Z flip)
    - AMASS npz    (`poses` key)    → `:z`
  Override with `up=:y` when loading from a Y-up smplcodec v2 dataset, e.g.
  `AMASS_SMPLX_NEUTRAL_Yup_smplFormat` (the root pose encodes only facing direction,
  not the Y→Z world reorientation).

`pivot_labels_path`: optional path to a `.npy` file with pivot joint probability
scores (see `load_pivot_labels`). When provided, `seq.pivot_joints` is populated.

# Example
```julia
seq = load_motion("motion.smpl")
seq = load_motion("/path/to/AMASS/ACCAD/Female1General/walk.npz")
seq = load_motion("walk.smpl"; pivot_labels_path="walk_stageii.npy")
seq = load_motion("walk_yup.smpl"; up=:y)                    # Y-up smplcodec v2
```
"""
function load_motion(path::String;
                     pivot_labels_path::Union{Nothing,String} = nothing,
                     up::Union{Nothing,Symbol}                = nothing
                     ) :: MotionSequence{Float32}
    d = NPZ.npzread(path)

    # ---- poses ----
    if haskey(d, "fullpose")
        # smplcodec v1: (N_frames, N_j, 3) — reshape to flat (N_frames, N_j*3)
        # permutedims (1,3,2) converts (N,J,D)→(N,D,J) so Julia's column-major
        # reshape produces [j0_x,j0_y,j0_z,j1_x,...] matching NumPy C-order.
        fp = d["fullpose"]                                              # (N, N_j, 3)
        N_frames, _, _ = size(fp)
        poses = reshape(permutedims(Float32.(fp), (1, 3, 2)), N_frames, :)  # (N, N_j*3)

    elseif haskey(d, "bodyPose")
        # smplcodec v2 (SMPLX): separate body/head/hand parts stored per component.
        # Joint order: body(0-21) | head(22-24) | leftHand(25-39) | rightHand(40-54)
        # Each part is (N, J_part, 3) — apply permutedims fix before reshape.
        N_frames = size(d["bodyPose"], 1)
        body  = reshape(permutedims(Float32.(d["bodyPose"]),     (1,3,2)), N_frames, :)  # (N, 66)
        head  = reshape(permutedims(Float32.(d["headPose"]),     (1,3,2)), N_frames, :)  # (N, 9)
        lhand = reshape(permutedims(Float32.(d["leftHandPose"]), (1,3,2)), N_frames, :)  # (N, 45)
        rhand = reshape(permutedims(Float32.(d["rightHandPose"]),(1,3,2)), N_frames, :)  # (N, 45)
        poses = hcat(body, head, lhand, rhand)                                           # (N, 165)

    elseif haskey(d, "poses")
        # AMASS format: already flat (N_frames, pose_dim)
        poses = Float32.(d["poses"])                             # (N_frames, pose_dim)

    else
        error("load_motion: unrecognised file format — expected 'fullpose', 'bodyPose', or 'poses' key in $path")
    end

    # ---- translation ----
    transl_key = haskey(d, "bodyTranslation") ? "bodyTranslation" :
                 haskey(d, "transl")           ? "transl"          : "trans"
    trans = Float32.(d[transl_key])                              # (N_frames, 3)

    # ---- shape parameters ----
    betas = if haskey(d, "betas")
        raw = d["betas"]
        ndims(raw) == 2 ? Float32.(raw[1, :]) : Float32.(raw)   # (N_b,)
    else
        zeros(Float32, 10)   # smplcodec v2 omits betas — use neutral shape
    end

    # ---- fps ----
    fps_val = if haskey(d, "frameRate")          # smplcodec v2
        Float32(d["frameRate"])
    elseif haskey(d, "fps")                      # smplcodec v1
        Float32(d["fps"])
    elseif haskey(d, "mocap_framerate")          # AMASS
        Float32(d["mocap_framerate"])
    else
        30.0f0
    end

    # ---- gender ----
    gender = if !haskey(d, "gender")
        :neutral
    elseif isa(d["gender"], Integer)
        # smplcodec v2: Int32 — 1=female, 2=male, anything else=neutral
        d["gender"] == 1 ? :female : d["gender"] == 2 ? :male : :neutral
    else
        _parse_gender(d["gender"])
    end

    # ---- model type ----
    pose_dim   = size(poses, 2)
    model_type = _infer_model_type(pose_dim)

    # ---- up axis ----
    # smplcodec v1 (fullpose key): root rotation does NOT encode a Y→Z world flip,
    #   so smpl_lbs output has Y as height → :y.
    # smplcodec v2 (bodyPose key) and AMASS npz (poses key): root rotation encodes
    #   the Y→Z flip → smpl_lbs output has Z as height → :z.
    # The `up` keyword overrides auto-detection, e.g. for Yup smplcodec v2 datasets
    #   (AMASS_SMPLX_NEUTRAL_Yup_smplFormat) where the root pose does not include the
    #   Y→Z flip and Y is the height axis.
    up_axis = if !isnothing(up)
        up                      # explicit caller override
    elseif haskey(d, "fullpose")
        :y                      # smplcodec v1 — Y-up convention
    else
        :z                      # smplcodec v2 + AMASS — Z-up world coords
    end

    pivot_joints = isnothing(pivot_labels_path) ? nothing :
                   load_pivot_labels(pivot_labels_path)
    if !isnothing(pivot_joints) && size(pivot_joints, 1) != size(poses, 1)
        @warn "load_motion: pivot_labels frame count ($(size(pivot_joints,1))) " *
              "≠ pose frame count ($(size(poses,1)))"
    end
    return MotionSequence{Float32}(poses, betas, trans, fps_val, model_type, gender,
                                   up_axis, pivot_joints)
end
