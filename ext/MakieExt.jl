# ext/MakieExt.jl — Visualization extension for SMPL.jl.
#
# Loaded automatically when any Makie backend is in the environment:
#   using SMPL, GLMakie    — opens interactive windows
#   using SMPL, CairoMakie — headless rendering (no display required)
#   using SMPL, WGLMakie   — browser-based (Jupyter/Pluto)
#
# Public API (all exported from the SMPL module):
#   bake_motion(model, seq)                 -> Array{Float32,3} (N_v, 3, N_frames)
#   viz_motion(model, seq; kwargs...)       -> Makie.Figure  (interactive player)
#   viz_motions(model, seqs; kwargs...)     -> Makie.Figure  (multi-motion player)
#   record_motion(model, seq, file; kwargs) -> nothing       (headless video)
#   render_frame(model, seq, frame, file; kwargs) -> nothing (headless PNG)
#
# Interactive player layout (viz_motion):
#   ┌─────────────────────────────────┐
#   │         LScene (3D view)         │
#   ├────────┬────────────────┬────────┤
#   │ [Play] │ ════════●═════ │  1.0x  │
#   │        │  frame slider  │ speed  │
#   └────────┴────────────────┴────────┘
#
# viz_motions(layout=:sidebyside): one LScene per motion, shared slider + play/pause
# viz_motions(layout=:overlay):    single LScene, each motion a distinct mesh color

module MakieExt

using SMPL
using Makie
using LinearAlgebra: I


# ---------------------------------------------------------------------------
# Internal helpers — coordinate system and scene setup
# ---------------------------------------------------------------------------

# Returns a 3×3 rotation matrix that converts from the sequence's up convention
# to Z-up (Makie's LScene camera default).
#   :z → identity  (data and Makie are both Z-up — no rotation needed)
#   :y → rotate Y-up to Z-up: maps (x,y,z)→(x,-z,y) so +Y (height) becomes +Z
function _up_rotation(up::Symbol) :: Matrix{Float32}
    up == :z && return Matrix{Float32}(I, 3, 3)
    # :y — rotate Y-up to Z-up: maps (x,y,z)→(x,-z,y) so +Y height becomes +Z in Makie
    # verts * R' with R=[1 0 0; 0 0 -1; 0 1 0] gives R'=[1 0 0; 0 0 1; 0 -1 0]
    # and [x,y,z]*R' = [x, -z, y] — Y column maps to output Z. ✓
    return Float32[1 0 0; 0 0 -1; 0 1 0]
end

# Compute a Rect3f that tightly contains the entire motion trajectory (after
# up-rotation), with enough margin to show the full body + some ground below.
# After rotation, Z is always the vertical (height) axis for Makie.
function _motion_rect(seq::SMPL.MotionSequence, R::Matrix{Float32}) :: Rect3f
    trans_rot = seq.trans * R'                      # (N, 3) rotated translations
    lo = vec(minimum(trans_rot, dims=1)) .- Float32[1.2, 1.2, 0.0]
    hi = vec(maximum(trans_rot, dims=1)) .+ Float32[1.2, 1.2, 2.2]
    lo[3] = min(lo[3], -0.1f0)                      # always include slightly below ground
    Rect3f(Vec3f(lo...), Vec3f((hi .- lo)...))
end

# Draw an opaque ground plane + grid in the XY plane at z=floor level.
# Makie's LScene camera is Z-up, so the XY plane (z=constant) is the horizontal floor.
#
# Two-layer approach:
#   1. Opaque white quad at motion-area bounds drawn first — occludes body geometry
#      that penetrates below z=0 (feet, crouching).  The Figure background is also
#      white (default), so the floor edge is invisible and the floor appears infinite:
#      white floor blends seamlessly into white sky.
#   2. Grid lines at the same bounds — in perspective these converge toward the horizon
#      giving a depth cue without expanding the scene limits.
function _ground_plane!(scene, rect::Rect3f; spacing=0.5f0)
    o = rect.origin;  w = rect.widths
    xmin, xmax = o[1], o[1] + w[1]
    ymin, ymax = o[2], o[2] + w[2]
    z0  = o[3]   # ground level (Z is up in Makie)

    # Layer 1 — opaque white floor quad at motion bounds.
    # Since the Figure background is white, the floor-to-background boundary is
    # invisible, so the floor appears to extend infinitely.
    poly!(scene,
          [Point3f(xmin, ymin, z0), Point3f(xmax, ymin, z0),
           Point3f(xmax, ymax, z0), Point3f(xmin, ymax, z0)];
          color = :white, strokewidth = 0f0)

    # Layer 2 — grid lines at rect bounds.
    # Keeping lines within bounds prevents Makie from expanding the scene limits
    # (which would zoom the camera out). The white background beyond the floor edges
    # gives the "infinite floor" look.
    for x in xmin:spacing:xmax
        lines!(scene, [Point3f(x, ymin, z0), Point3f(x, ymax, z0)];
               color = (:gray, 0.4f0), linewidth = 0.5f0)
    end
    for y in ymin:spacing:ymax
        lines!(scene, [Point3f(xmin, y, z0), Point3f(xmax, y, z0)];
               color = (:gray, 0.4f0), linewidth = 0.5f0)
    end
end

# Draw an RGB axis marker at world origin (0,0,0).
# X=red, Y=green, Z=blue — matches Blender / Unreal Engine convention.
#
# Only draws if world origin falls within `rect`.  Geometry added outside the
# scene rect triggers Makie's auto-limit expansion which zooms the camera out;
# when the body has walked far from origin the marker would be off-screen anyway.
function _origin_marker!(scene, rect::Rect3f; len::Float32=0.35f0)
    o = rect.origin
    # Check whether (0,0,0) is inside rect — if not, skip drawing to avoid
    # Makie expanding scene limits beyond the body bounds.
    for i in 1:3
        (o[i] <= 0f0 <= o[i] + rect.widths[i]) || return
    end
    origins    = [Point3f(0,0,0), Point3f(0,0,0), Point3f(0,0,0)]
    directions = [Vec3f(len,0,0), Vec3f(0,len,0), Vec3f(0,0,len)]
    colors     = [:red, :green, :blue]
    arrows3d!(scene, origins, directions;
              color       = colors,
              shaftradius = 0.012f0,
              tipradius   = 0.025f0,
              tiplength   = 0.07f0)
end


# Position the Camera3D of `lscene` using explicit world-space eye/lookat/upvector
# and a vertical FOV.  Must be called AFTER all geometry is added so that Makie's
# auto-limit expansion (triggered by poly!, mesh!, etc.) cannot override it.
function _setup_camera!(
    lscene,
    rect  :: Rect3f;
    camera_eye      :: Union{Nothing, Vec3f} = nothing,
    camera_lookat   :: Union{Nothing, Vec3f} = nothing,
    camera_upvector :: Vec3f                 = Vec3f(0f0, 0f0, 1f0),
    camera_fov      :: Float32               = 45f0
)
    o  = rect.origin
    cx = o[1] + rect.widths[1] / 2f0   # horizontal trajectory centre
    cy = o[2] + rect.widths[2] / 2f0
    z0 = o[3]                            # floor Z level

    # Default: nearly-horizontal 3/4-front view ~2.5 m from body, body fills ~2/3 viewport.
    # Camera at chest/shoulder height looking slightly down at the pelvis — minimal
    # floor distortion, body appears centred in the image.
    def_lookat = Vec3f(cx, cy, z0 + 0.9f0)               # approx pelvis height
    def_eye    = Vec3f(cx + 1.5f0, cy - 2.0f0, z0 + 1.2f0)  # ≈2.6 m, 7° tilt down

    eye    = isnothing(camera_eye)    ? def_eye    : camera_eye
    lookat = isnothing(camera_lookat) ? def_lookat : camera_lookat

    Makie.update_cam!(lscene.scene, eye, lookat, camera_upvector)
    Makie.cameracontrols(lscene.scene).fov[] = camera_fov
    return nothing
end


# ---------------------------------------------------------------------------
# _pivot_markers! — add pivot probability markers (Observable path)
# ---------------------------------------------------------------------------

# Add pivot probability sphere markers to `scene`, reactive to `frame_obs`.
# `joints_obs` is an Observable{Matrix{Float32}} of shape (N_j, 3) (already rotated).
# Returns nothing if pivot_mode == :none or seq.pivot_joints === nothing.
#
# Design: always emits exactly N_tracked points to keep Observable vector length
# constant across frame updates (Makie requirement). Non-highlighted joints use
# alpha=0 (transparent).
#   :max mode       — single vivid red sphere at the highest-scoring joint.
#   :threshold mode — :plasma colormap (dark-purple→orange→yellow); all values
#                     visible on the grey body mesh.
# Joint positions are always within the body, so they fall inside the motion rect —
# no Makie auto-limit expansion issue.
function _pivot_markers!(
    scene,
    frame_obs            :: Observable{Int},
    joints_obs           :: Observable{Matrix{Float32}},
    seq                  :: SMPL.MotionSequence,
    pivot_mode           :: Symbol,
    pivot_threshold      :: Float32,
    pivot_joint_indices  :: AbstractVector{Int},
)
    (pivot_mode == :none || isnothing(seq.pivot_joints)) && return

    labels    = seq.pivot_joints          # (N_frames, N_tracked)
    N_frames  = size(labels, 1)
    N_tracked = size(labels, 2)
    cmap      = Makie.to_colormap(:plasma)   # used only for :threshold mode

    # colors_obs: (N_tracked,) RGBAf — alpha=0 hides non-highlighted joints
    colors_obs = @lift begin
        row  = labels[min($frame_obs, N_frames), :]   # (N_tracked,) scores for this frame
        best = argmax(row)
        if pivot_mode == :max
            # Fixed vivid red — clearly visible on the grey body mesh
            [RGBAf(1f0, 0f0, 0f0, k == best ? 1f0 : 0f0) for k in 1:N_tracked]
        else  # :threshold
            probs = clamp.(row, 0f0, 1f0)
            cols  = map(p -> Makie.interpolated_getindex(cmap, p), probs)
            [RGBAf(cols[k].r, cols[k].g, cols[k].b,
                   row[k] >= pivot_threshold ? 1f0 : 0f0) for k in 1:N_tracked]
        end
    end

    # positions_obs: always emit all N_tracked joint positions (fixed-length vector)
    positions_obs = @lift begin
        jnts = $joints_obs   # (N_j, 3)
        [Point3f(jnts[j, 1], jnts[j, 2], jnts[j, 3]) for j in pivot_joint_indices]
    end

    meshscatter!(scene, positions_obs;
                 markersize = 0.05f0, color = colors_obs)
    return nothing
end


# ---------------------------------------------------------------------------
# _render_pivot_markers! — add pivot probability markers (non-Observable path)
# ---------------------------------------------------------------------------

# Non-reactive version for render_frame: takes a concrete SMPLOutput and frame index.
# Draws meshscatter directly with plain vectors (no Observable wiring needed).
# Color scheme matches _pivot_markers!: red for :max, :plasma for :threshold.
function _render_pivot_markers!(
    scene,
    out                  :: SMPL.SMPLOutput,
    seq                  :: SMPL.MotionSequence,
    frame                :: Int,
    R                    :: Matrix{Float32},
    pivot_mode           :: Symbol,
    pivot_threshold      :: Float32,
    pivot_joint_indices  :: AbstractVector{Int},
)
    labels    = seq.pivot_joints          # (N_frames, N_tracked)
    N_tracked = size(labels, 2)
    row       = labels[frame, :]          # (N_tracked,) scores
    joints    = Array(out.joints) * R'    # (N_j, 3)
    best      = argmax(row)

    pts = [Point3f(joints[j, 1], joints[j, 2], joints[j, 3]) for j in pivot_joint_indices]

    colors = if pivot_mode == :max
        # Fixed vivid red — clearly visible on the grey body mesh
        [RGBAf(1f0, 0f0, 0f0, k == best ? 1f0 : 0f0) for k in 1:N_tracked]
    else  # :threshold
        probs = clamp.(row, 0f0, 1f0)
        cmap  = Makie.to_colormap(:plasma)
        cols  = map(p -> Makie.interpolated_getindex(cmap, p), probs)
        [RGBAf(cols[k].r, cols[k].g, cols[k].b,
               row[k] >= pivot_threshold ? 1f0 : 0f0) for k in 1:N_tracked]
    end

    meshscatter!(scene, pts; markersize = 0.05f0, color = colors)
    return nothing
end


# ---------------------------------------------------------------------------
# bake_motion — pre-compute all vertices for a motion sequence
# ---------------------------------------------------------------------------

"""
    bake_motion(model, seq::MotionSequence) -> Array{Float32,3}

Run `smpl_lbs` for every frame in `seq` and return a pre-computed vertex array
of shape `(N_v, 3, N_frames)`.

Baking is useful when you need random-access playback (scrubbing the timeline
backward) or when rendering many frames headlessly without re-computing LBS at
each frame interactively.

For very long sequences (> a few thousand frames) the array can be large (SMPL:
~6890 × 3 × 4 bytes × N_frames). Stream with `smpl_lbs` per-frame if memory is
a concern.
"""
function SMPL.bake_motion(model, seq::SMPL.MotionSequence{T}) where {T}
    N_frames  = size(seq.poses, 1)
    N_v       = size(model.v_template, 1)
    verts_all = Array{T}(undef, N_v, 3, N_frames)

    betas = seq.betas
    @inbounds for i in 1:N_frames
        θ     = seq.poses[i, :]           # (pose_dim,)
        trans = seq.trans[i, :]           # (3,)
        out   = smpl_lbs(model, betas, θ, trans)
        verts_all[:, :, i] .= Array(out.vertices)  # (N_v, 3) — materialise if GPU
    end
    return verts_all
end


# ---------------------------------------------------------------------------
# _make_player_controls — shared UI widgets for interactive players
# ---------------------------------------------------------------------------

# Returns (fig, scenes, frame_obs, play_status, slider) ready for callers to
# wire mesh/skeleton observables against `frame_obs`.
# `rect` is used to initialise the scene camera bounds.
function _make_player_controls(fig, N_frames::Int, n_scenes::Int, rect::Rect3f)
    # Scene row
    scenes = [LScene(fig[1, k]; show_axis = false, scenekw = (; limits = rect))
              for k in 1:n_scenes]

    # Control row
    play_status = Observable("▶ Play")
    btn         = Button(fig[2, 1]; label = play_status, width = 80)
    slider      = Slider(fig[2, 2:max(2, n_scenes)];
                         range = 1:N_frames, startvalue = 1)
    speed_menu  = Menu(fig[2, max(2, n_scenes)+1];
                       options = ["0.25×", "0.5×", "1×", "2×", "4×"],
                       default = "1×", width = 70)

    speed_map = Dict("0.25×" => 4.0, "0.5×" => 2.0, "1×" => 1.0,
                     "2×" => 0.5,   "4×" => 0.25)

    # Play/pause logic — advances slider asynchronously
    on(btn.clicks) do _
        if play_status[] == "▶ Play"
            play_status[] = "⏸ Pause"
            @async while play_status[] == "⏸ Pause" &&
                          slider.value[] < N_frames
                set_close_to!(slider, slider.value[] + 1)
                sleep(speed_map[speed_menu.selection[]] / 30.0)
            end
            # Auto-stop at last frame
            if slider.value[] >= N_frames
                play_status[] = "▶ Play"
            end
        else
            play_status[] = "▶ Play"
        end
    end

    return (fig, scenes, slider.value, play_status, slider)
end


# ---------------------------------------------------------------------------
# viz_motion — interactive single-motion player
# ---------------------------------------------------------------------------

"""
    viz_motion(model, seq::MotionSequence; show_skeleton=false,
               pivot_mode=:none, pivot_threshold=0.5f0, pivot_joint_indices=1:23,
               figure_kwargs...) -> Figure

Open an interactive motion player window. The figure contains:
  - A 3D scene showing the body mesh (and optionally the skeleton).
  - A timeline slider for scrubbing.
  - A play/pause button with configurable speed (0.25×, 0.5×, 1×, 2×, 4×).

The scene is automatically sized to fit the motion trajectory. A ground-plane
grid is drawn at foot level. The `seq.up` field controls the coordinate system:
`:y` (default) renders as-is; `:z` applies a −90° X rotation so Z-up data
appears upright.

Requires a display-capable backend (GLMakie). For headless rendering use
`record_motion` or `render_frame` with CairoMakie.

# Arguments
- `model`: a `BodyModel` or `SUPRModel` (CPU or GPU).
- `seq`:   a `MotionSequence` loaded with `load_motion`.
- `show_skeleton`: overlay joint positions as spheres (default: false).
- `pivot_mode`: `:none` (default), `:max` (highest-prob joint per frame), or
  `:threshold` (all joints ≥ `pivot_threshold`). Requires `seq.pivot_joints`.
- `pivot_threshold`: score threshold for `:threshold` mode (default: 0.5).
- `pivot_joint_indices`: 1-indexed joint indices in the model that the columns
  of `seq.pivot_joints` map to (default: `1:23` for SMPLX body joints).
- `figure_kwargs`: forwarded to `Makie.Figure(...)`.

# Example
```julia
using SMPL, GLMakie
model = create_smplx_neutral()
seq   = load_motion("walk.smpl"; pivot_labels_path="walk_stageii.npy")
fig   = viz_motion(model, seq; pivot_mode=:max)
```
"""
function SMPL.viz_motion(model, seq::SMPL.MotionSequence;
                         show_skeleton        :: Bool                  = false,
                         pivot_mode           :: Symbol                = :none,
                         pivot_threshold      :: Float32               = 0.5f0,
                         pivot_joint_indices  :: AbstractVector{Int}   = 1:23,
                         camera_eye           :: Union{Nothing, Vec3f} = nothing,
                         camera_lookat        :: Union{Nothing, Vec3f} = nothing,
                         camera_upvector      :: Vec3f                 = Vec3f(0f0, 0f0, 1f0),
                         camera_fov           :: Float32               = 45f0,
                         figure_kwargs...)
    N_frames = size(seq.poses, 1)
    R    = _up_rotation(seq.up)
    rect = _motion_rect(seq, R)

    # Pre-compute first frame to get faces
    out0  = smpl_lbs(model, seq.betas, seq.poses[1, :], seq.trans[1, :])
    faces = out0.faces   # (N_f, 3) UInt32 1-indexed

    fig = Figure(; figure_kwargs...)
    _, scenes, frame_obs, _, _ = _make_player_controls(fig, N_frames, 1, rect)
    scene = scenes[1]
    _ground_plane!(scene.scene, rect)
    _origin_marker!(scene.scene, rect)

    # Single smpl_lbs call per frame — shared by mesh, skeleton, and pivot markers
    lbs_obs = @lift smpl_lbs(model, seq.betas, seq.poses[$frame_obs, :],
                              seq.trans[$frame_obs, :])

    verts_obs = @lift Array($lbs_obs.vertices) * R'   # (N_v, 3) rotated
    mesh!(scene, verts_obs, faces; color = :lightgray, shading = true)

    need_joints = show_skeleton || (pivot_mode != :none && !isnothing(seq.pivot_joints))
    if need_joints
        joints_obs = @lift Matrix{Float32}(Array($lbs_obs.joints) * R')   # (N_j, 3)
        show_skeleton && meshscatter!(scene, joints_obs;
                                      markersize = 0.02f0, color = :red)
        _pivot_markers!(scene.scene, frame_obs, joints_obs, seq,
                        pivot_mode, pivot_threshold, pivot_joint_indices)
    end

    _setup_camera!(scene, rect;
                   camera_eye=camera_eye, camera_lookat=camera_lookat,
                   camera_upvector=camera_upvector, camera_fov=camera_fov)

    display(fig)
    return fig
end


# ---------------------------------------------------------------------------
# viz_motions — multi-motion player
# ---------------------------------------------------------------------------

"""
    viz_motions(model, seqs::AbstractVector{<:MotionSequence};
                layout::Symbol = :sidebyside,
                labels::AbstractVector{String} = String[],
                figure_kwargs...) -> Figure

Display multiple motions simultaneously with a shared timeline.

# Layout options
- `:sidebyside` — one 3D panel per motion in a horizontal grid. All panels
  are driven by the same frame slider, so they stay synchronised.
- `:overlay` — all motions rendered in a single 3D panel using different
  mesh colors (from a categorical colormap).

# Arguments
- `model`: shared body model used for all sequences.
- `seqs`:  vector of `MotionSequence` objects (can have different lengths;
           the slider range covers the longest one).
- `labels`: optional title labels per motion (`:sidebyside` only).

# Example
```julia
using SMPL, GLMakie
model = create_smplx_neutral()
seqs  = [load_motion("walk.smpl"), load_motion("run.smpl")]
viz_motions(model, seqs; layout=:sidebyside, labels=["Walk", "Run"])
```
"""
function SMPL.viz_motions(model, seqs::AbstractVector{<:SMPL.MotionSequence};
                          layout          :: Symbol                  = :sidebyside,
                          labels          :: AbstractVector{String}  = String[],
                          camera_eye      :: Union{Nothing, Vec3f}   = nothing,
                          camera_lookat   :: Union{Nothing, Vec3f}   = nothing,
                          camera_upvector :: Vec3f                   = Vec3f(0f0, 0f0, 1f0),
                          camera_fov      :: Float32                 = 45f0,
                          figure_kwargs...)
    N_seq    = length(seqs)
    N_frames = maximum(s -> size(s.poses, 1), seqs)

    # Faces are shared (same model for all sequences)
    faces = smpl_lbs(model, seqs[1].betas, seqs[1].poses[1,:], seqs[1].trans[1,:]).faces

    # Use up axis from first sequence for scene setup
    R    = _up_rotation(seqs[1].up)
    rect = _motion_rect(seqs[1], R)

    # Color palette for overlay mode
    palette = Makie.to_colormap(:tab10, max(N_seq, 2))

    fig = Figure(; figure_kwargs...)

    if layout == :sidebyside
        n_panels = N_seq
        _, scenes, frame_obs, _, _ = _make_player_controls(fig, N_frames, n_panels, rect)

        for k in 1:N_seq
            lbl = k <= length(labels) ? labels[k] : "Motion $k"
            Label(fig[0, k], lbl; fontsize = 14, tellwidth = false)
            _ground_plane!(scenes[k].scene, rect)
            _origin_marker!(scenes[k].scene, rect)
        end

        for (k, seq) in enumerate(seqs)
            scene  = scenes[k]
            R_seq  = _up_rotation(seq.up)
            verts_obs = @lift begin
                i   = min($frame_obs, size(seq.poses, 1))
                out = smpl_lbs(model, seq.betas, seq.poses[i,:], seq.trans[i,:])
                Array(out.vertices) * R_seq'
            end
            mesh!(scene, verts_obs, faces;
                  color = palette[k], shading = true)
            _setup_camera!(scene, rect;
                           camera_eye=camera_eye, camera_lookat=camera_lookat,
                           camera_upvector=camera_upvector, camera_fov=camera_fov)
        end

    elseif layout == :overlay
        _, scenes, frame_obs, _, _ = _make_player_controls(fig, N_frames, 1, rect)
        scene = scenes[1]
        _ground_plane!(scene.scene, rect)
        _origin_marker!(scene.scene, rect)

        for (k, seq) in enumerate(seqs)
            R_seq = _up_rotation(seq.up)
            verts_obs = @lift begin
                i   = min($frame_obs, size(seq.poses, 1))
                out = smpl_lbs(model, seq.betas, seq.poses[i,:], seq.trans[i,:])
                Array(out.vertices) * R_seq'
            end
            mesh!(scene, verts_obs, faces;
                  color = (palette[k], 0.7f0),   # 70% opaque for overlap visibility
                  shading = true)
        end

        _setup_camera!(scenes[1], rect;
                       camera_eye=camera_eye, camera_lookat=camera_lookat,
                       camera_upvector=camera_upvector, camera_fov=camera_fov)

    else
        error("viz_motions: unknown layout=$layout — use :sidebyside or :overlay")
    end

    display(fig)
    return fig
end


# ---------------------------------------------------------------------------
# record_motion — headless video export
# ---------------------------------------------------------------------------

"""
    record_motion(model, seq::MotionSequence, outfile::String;
                  fps::Real = seq.fps,
                  resolution::Tuple{Int,Int} = (1920, 1080),
                  show_skeleton::Bool = false,
                  pivot_mode=:none, pivot_threshold=0.5f0,
                  pivot_joint_indices=1:23) -> nothing

Render all frames of `seq` and write a video file (`.mp4`, `.mkv`, `.gif`).

Works with any Makie backend:
  - **CairoMakie** — fully headless, no display or GPU required (recommended
    for servers / CI pipelines).
  - **GLMakie** — uses an offscreen framebuffer; faster on a machine with a GPU.

The output file format is determined by the extension of `outfile`.
Pivot kwargs: see `viz_motion` for pivot visualization options.

# Example
```julia
using SMPL, CairoMakie
model = create_smplx_neutral()
seq   = load_motion("walk.smpl"; pivot_labels_path="walk_stageii.npy")
record_motion(model, seq, "walk.mp4"; fps=30, pivot_mode=:max)
```
"""
function SMPL.record_motion(model, seq::SMPL.MotionSequence, outfile::String;
                             fps                  :: Real                  = seq.fps,
                             resolution           :: Tuple{Int,Int}        = (1920, 1080),
                             show_skeleton        :: Bool                  = false,
                             pivot_mode           :: Symbol                = :none,
                             pivot_threshold      :: Float32               = 0.5f0,
                             pivot_joint_indices  :: AbstractVector{Int}   = 1:23,
                             camera_eye           :: Union{Nothing, Vec3f} = nothing,
                             camera_lookat        :: Union{Nothing, Vec3f} = nothing,
                             camera_upvector      :: Vec3f                 = Vec3f(0f0, 0f0, 1f0),
                             camera_fov           :: Float32               = 45f0)
    N_frames = size(seq.poses, 1)
    faces    = smpl_lbs(model, seq.betas, seq.poses[1,:], seq.trans[1,:]).faces
    R        = _up_rotation(seq.up)
    rect     = _motion_rect(seq, R)

    frame_obs = Observable(1)
    fig = Figure(; size = resolution)
    scene = LScene(fig[1, 1]; show_axis = false, scenekw = (; limits = rect))
    _ground_plane!(scene.scene, rect)
    _origin_marker!(scene.scene, rect)

    # Single smpl_lbs call per frame — shared by mesh, skeleton, and pivot markers
    lbs_obs   = @lift smpl_lbs(model, seq.betas, seq.poses[$frame_obs,:],
                                seq.trans[$frame_obs,:])
    verts_obs = @lift Array($lbs_obs.vertices) * R'
    mesh!(scene, verts_obs, faces; color = :lightgray, shading = true)

    need_joints = show_skeleton || (pivot_mode != :none && !isnothing(seq.pivot_joints))
    if need_joints
        joints_obs = @lift Matrix{Float32}(Array($lbs_obs.joints) * R')   # (N_j, 3)
        show_skeleton && meshscatter!(scene, joints_obs;
                                      markersize = 0.02f0, color = :red)
        _pivot_markers!(scene.scene, frame_obs, joints_obs, seq,
                        pivot_mode, pivot_threshold, pivot_joint_indices)
    end

    _setup_camera!(scene, rect;
                   camera_eye=camera_eye, camera_lookat=camera_lookat,
                   camera_upvector=camera_upvector, camera_fov=camera_fov)

    Makie.record(fig, outfile, 1:N_frames; framerate = Int(round(fps))) do i
        frame_obs[] = i
    end
    return nothing
end


# ---------------------------------------------------------------------------
# render_frame — single-frame headless PNG export
# ---------------------------------------------------------------------------

"""
    render_frame(model, seq::MotionSequence, frame::Int, outfile::String;
                 resolution::Tuple{Int,Int} = (1920, 1080),
                 show_skeleton::Bool = false,
                 pivot_mode=:none, pivot_threshold=0.5f0,
                 pivot_joint_indices=1:23) -> nothing

Render a single frame of `seq` and save it as a PNG (or any format supported
by Makie.save, such as `.svg` when using CairoMakie).

`frame` is 1-indexed (1 = first frame, `size(seq.poses,1)` = last frame).

The `seq.up` field is used to orient the scene: `:z`-up sequences are
automatically rotated so the body appears upright in the output image.
Pivot kwargs: see `viz_motion` for pivot visualization options.

# Example
```julia
using SMPL, CairoMakie
render_frame(model, seq, 42, "frame042.png"; resolution=(1280, 720))
render_frame(model, seq, 42, "pivot042.png"; pivot_mode=:max)
```
"""
function SMPL.render_frame(model, seq::SMPL.MotionSequence, frame::Int, outfile::String;
                            resolution           :: Tuple{Int,Int}        = (1920, 1080),
                            show_skeleton        :: Bool                  = false,
                            pivot_mode           :: Symbol                = :none,
                            pivot_threshold      :: Float32               = 0.5f0,
                            pivot_joint_indices  :: AbstractVector{Int}   = 1:23,
                            camera_eye           :: Union{Nothing, Vec3f} = nothing,
                            camera_lookat        :: Union{Nothing, Vec3f} = nothing,
                            camera_upvector      :: Vec3f                 = Vec3f(0f0, 0f0, 1f0),
                            camera_fov           :: Float32               = 45f0)
    R     = _up_rotation(seq.up)
    # Use a single-frame sequence for bounds so the scene is centred on this frame,
    # not stretched to cover the entire trajectory (which would make the body tiny).
    pj_slice = isnothing(seq.pivot_joints) ? nothing :
               seq.pivot_joints[frame:frame, :]   # (1, N_tracked)
    single = SMPL.MotionSequence(seq.poses[frame:frame,:], seq.betas,
                                 seq.trans[frame:frame,:], seq.fps,
                                 seq.model_type, seq.gender, seq.up, pj_slice)
    rect  = _motion_rect(single, R)
    faces = smpl_lbs(model, seq.betas, seq.poses[1,:], seq.trans[1,:]).faces
    out   = smpl_lbs(model, seq.betas, seq.poses[frame,:], seq.trans[frame,:])

    fig   = Figure(; size = resolution)
    scene = LScene(fig[1, 1]; show_axis = false, scenekw = (; limits = rect))
    _ground_plane!(scene.scene, rect)
    _origin_marker!(scene.scene, rect)

    mesh!(scene, Array(out.vertices) * R', faces;
          color = :lightgray, shading = true)

    if show_skeleton
        meshscatter!(scene, Array(out.joints) * R';
                     markersize = 0.02f0, color = :red)
    end

    if pivot_mode != :none && !isnothing(seq.pivot_joints)
        _render_pivot_markers!(scene.scene, out, seq, frame, R,
                               pivot_mode, pivot_threshold, pivot_joint_indices)
    end

    _setup_camera!(scene, rect;
                   camera_eye=camera_eye, camera_lookat=camera_lookat,
                   camera_upvector=camera_upvector, camera_fov=camera_fov)

    Makie.save(outfile, fig)
    return nothing
end


end  # module MakieExt
