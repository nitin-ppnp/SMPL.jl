# SMPL.jl — Julia implementation of the SMPL family of human body models.
#
# Supported models:
#   SMPL    — 24 joints, 6890 vertices   (Loper et al., 2015)
#   SMPLX   — 55 joints, 10475 vertices  (Pavlakos et al., 2019)
#   SUPR    — 75 joints, 10475 vertices  (Osman et al., 2022)
#
# Core modules:
#   types.jl      — BodyModel{T,A2}, SUPRModel{T,A2}, SMPLOutput{T,A2}, MotionSequence{T}
#   math.jl       — rodrigues, quat_feat, forward_kinematics
#   io.jl         — create_smpl/smplx/supr + DataDeps __init__ + _read_credentials
#   model.jl      — smpl_lbs, pivot_fk
#   motion_io.jl  — load_motion
#
# GPU support via Adapt.jl (weak dependency, see ext/AdaptExt.jl):
#   Adapt.adapt(CuArray, model)  moves the model to GPU
#   smpl_lbs then runs on GPU unchanged
#
# Visualization via Makie.jl (weak dependency, see ext/MakieExt.jl):
#   using SMPL, GLMakie    — interactive player (viz_motion, viz_motions)
#   using SMPL, CairoMakie — headless rendering (record_motion, render_frame)
#
# Static compilation (JuliaC trimmer path):
#   See staticSMPL.jl and static/static_io.jl for the .smplbin binary format.

module SMPL

using LinearAlgebra
using StaticArrays
using SparseArrays

include("types.jl")      # BodyModel, SUPRModel, SMPLOutput, MotionSequence
include("math.jl")       # rodrigues, quat_feat, forward_kinematics
include("io.jl")         # create_smpl*, create_smplx*, create_supr*, __init__
include("model.jl")      # smpl_lbs, _lbs_skin, pivot_fk
include("motion_io.jl")  # load_motion

# --- Structs ---
export BodyModel, SUPRModel, SMPLOutput, MotionSequence

# --- SMPL model constructors ---
export create_smpl, create_smpl_female, create_smpl_male, create_smpl_neutral

# --- SMPLX model constructors ---
export create_smplx, create_smplx_female, create_smplx_male, create_smplx_neutral

# --- SUPR model constructors ---
export create_supr, create_supr_female, create_supr_male, create_supr_neutral

# --- Forward pass ---
export smpl_lbs, pivot_fk

# --- Motion IO ---
export load_motion

# --- Visualization (populated by ext/MakieExt.jl when a Makie backend is loaded) ---

"""
    bake_motion(model, seq::MotionSequence) -> Array{Float32,3}

Run `smpl_lbs` for every frame in `seq` and return a pre-computed vertex array of
shape `(N_v, 3, N_frames)`. Use this to avoid repeated forward-pass overhead when
rendering or exporting a motion clip.

Requires a Makie backend (`GLMakie`, `CairoMakie`, or `WGLMakie`) to be loaded.
"""
function bake_motion end

"""
    viz_motion(model, seq::MotionSequence; show_skeleton=false,
               camera_eye=nothing, camera_lookat=nothing,
               camera_upvector=Vec3f(0,0,1), camera_fov=45f0, figure_kwargs...)

Open an interactive Makie window with a frame slider, play/pause button, and speed
control for the given motion sequence. Requires `GLMakie` or `WGLMakie`.

Camera kwargs: `camera_eye` and `camera_lookat` are `Vec3f` world-space positions;
`camera_fov` is the vertical field-of-view in degrees (default 45°).
"""
function viz_motion end

"""
    viz_motions(model, seqs; layout=:sidebyside, labels=nothing,
                camera_eye=nothing, camera_lookat=nothing,
                camera_upvector=Vec3f(0,0,1), camera_fov=45f0, figure_kwargs...)

Visualize multiple motion sequences simultaneously.

- `layout = :sidebyside` — one scene per motion, shared playback controls.
- `layout = :overlay` — all motions in a single scene, each tinted a distinct color.

Requires `GLMakie` or `WGLMakie`.
"""
function viz_motions end

"""
    record_motion(model, seq::MotionSequence, outfile::String;
                  fps=nothing, resolution=(1280,720), show_skeleton=false,
                  camera_eye=nothing, camera_lookat=nothing,
                  camera_upvector=Vec3f(0,0,1), camera_fov=45f0)

Render `seq` to a video file (`.mp4`, `.gif`, etc.) without a display. Uses
`Makie.record` — works headlessly with `CairoMakie`. `fps` defaults to `seq.fps`.
"""
function record_motion end

"""
    render_frame(model, seq::MotionSequence, frame::Int, outfile::String;
                 resolution=(1280,720), show_skeleton=false,
                 camera_eye=nothing, camera_lookat=nothing,
                 camera_upvector=Vec3f(0,0,1), camera_fov=45f0)

Render a single frame from `seq` to an image file (`.png`, `.svg`, etc.) without a
display. Works headlessly with `CairoMakie`.
"""
function render_frame end

export bake_motion, viz_motion, viz_motions, record_motion, render_frame

end  # module SMPL
