# Visualization

SMPL.jl visualization is provided by a package extension that loads automatically when any Makie backend is imported. The same API works across all backends.

## Backend Selection

| Backend | Use case |
|---------|----------|
| `GLMakie` | Interactive windows on a machine with a display and GPU |
| `CairoMakie` | Headless rendering — servers, CI, no display required |
| `WGLMakie` | Browser-based — Jupyter notebooks, Pluto |

```julia
# Interactive
using SMPL, GLMakie

# Headless / server
using SMPL, CairoMakie
```

## `bake_motion`

Pre-compute all frames before rendering. Useful for scrubbing backward or repeated rendering of the same motion.

```julia
verts = bake_motion(model, seq)   # Array{Float32,3} (N_v, 3, N_frames)
```

Memory estimate: `N_v × 3 × 4 bytes × N_frames`
- SMPL 1000 frames ≈ 79 MB
- SMPLX 1000 frames ≈ 120 MB

For streaming playback (memory-constrained), `viz_motion` recomputes LBS per frame without baking.

## `viz_motion` — Interactive Player

```julia
fig = viz_motion(model, seq)
fig = viz_motion(model, seq; show_skeleton = true)
fig = viz_motion(model, seq; figure_kwargs = (size = (1280, 720),))

# Custom camera
fig = viz_motion(model, seq;
                 camera_eye    = Vec3f(0, -4, 1.5),
                 camera_lookat = Vec3f(0, 0, 0.9),
                 camera_fov    = 35f0)
```

**Layout:**

```
┌──────────────────────────────────────────┐
│              LScene (3D view)             │
├──────────┬───────────────────────┬────────┤
│  ▶ Play  │  ══════════●═════════ │  1.0×  │
│          │       frame slider    │ speed  │
└──────────┴───────────────────────┴────────┘
```

**Keywords:**
- `show_skeleton::Bool` (default `false`) — overlays joint spheres on the mesh
- `camera_eye::Union{Nothing,Vec3f}` — camera position in world space. Default: ~2.5 m from body, 3/4-back view.
- `camera_lookat::Union{Nothing,Vec3f}` — world-space point the camera looks toward. Default: trajectory centre at pelvis height (~0.9 m above floor).
- `camera_upvector::Vec3f` — camera up direction. Default `Vec3f(0,0,1)` (Z-up world).
- `camera_fov::Float32` — vertical field-of-view in degrees. Default `45f0`. Lower = telephoto, higher = wide-angle.
- `figure_kwargs...` — forwarded to `Makie.Figure(...)`; use `size=(w,h)` to set resolution

## `viz_motions` — Multi-Motion Player

```julia
seqs = [load_motion("a.smpl"), load_motion("b.smpl"), load_motion("c.smpl")]

# One panel per motion, shared timeline
fig = viz_motions(model, seqs;
                  layout = :sidebyside,
                  labels = ["Walk", "Run", "Jump"])

# All motions overlaid with distinct colors
fig = viz_motions(model, seqs; layout = :overlay)

# Custom camera (applies to all panels in :sidebyside mode)
fig = viz_motions(model, seqs;
                  camera_eye    = Vec3f(0, -4, 1.5),
                  camera_lookat = Vec3f(0, 0, 0.9),
                  camera_fov    = 35f0)
```

**Keywords:**
- `layout::Symbol` — `:sidebyside` (default) or `:overlay`
- `labels::Vector{String}` — column titles for `:sidebyside` mode
- `camera_eye`, `camera_lookat`, `camera_upvector`, `camera_fov` — same as `viz_motion`; applied to all panels
- `figure_kwargs...` — forwarded to `Makie.Figure(...)`

The timeline slider and play/pause button are shared across all motions, so they stay synchronised. If sequences have different lengths, the slider covers the longest one; shorter sequences hold their last frame.

## `record_motion` — Headless Video

```julia
# Works with any backend; CairoMakie is recommended for headless
using SMPL, CairoMakie

record_motion(model, seq, "output.mp4";
              fps        = 30,
              resolution = (1920, 1080),
              show_skeleton = false)

# With custom camera
record_motion(model, seq, "output.mp4";
              camera_eye    = Vec3f(0, -4, 1.5),
              camera_lookat = Vec3f(0, 0, 0.9),
              camera_fov    = 28f0)
```

Output format is determined by the file extension. Supported formats depend on the backend:
- CairoMakie: `.mp4`, `.gif`
- GLMakie: `.mp4`, `.mkv`, `.gif`

**Keywords:**
- `fps::Real` — frame rate (default: `seq.fps`)
- `resolution::Tuple{Int,Int}` — output resolution in pixels
- `show_skeleton::Bool` — overlay joint spheres
- `camera_eye`, `camera_lookat`, `camera_upvector`, `camera_fov` — same as `viz_motion`

## `render_frame` — Single Frame

```julia
render_frame(model, seq, 42, "frame_042.png";
             resolution    = (1920, 1080),
             show_skeleton = false)

# Telephoto front view
render_frame(model, seq, 1, "frame.png";
             camera_eye    = Vec3f(0, -5, 1.5),
             camera_lookat = Vec3f(0, 0, 0.9),
             camera_fov    = 28f0)
```

`frame` is 1-indexed. CairoMakie supports `.png`, `.svg`, `.pdf` as output formats.

**Keywords:**
- `resolution::Tuple{Int,Int}` — output resolution in pixels
- `show_skeleton::Bool` — overlay joint spheres
- `camera_eye`, `camera_lookat`, `camera_upvector`, `camera_fov` — same as `viz_motion`

## Camera Control

All four visualization functions accept the same four camera kwargs:

| Kwarg | Type | Default | Description |
|---|---|---|---|
| `camera_eye` | `Union{Nothing,Vec3f}` | `nothing` | Camera position in world space. Default: ~2.5 m from body, 3/4-back view. |
| `camera_lookat` | `Union{Nothing,Vec3f}` | `nothing` | World-space point the camera looks toward. Default: trajectory centre at pelvis height (~0.9 m). |
| `camera_upvector` | `Vec3f` | `Vec3f(0,0,1)` | Camera up direction (Z-up world). |
| `camera_fov` | `Float32` | `45f0` | Vertical field-of-view in degrees. Lower = telephoto, higher = wide-angle. |

A named tuple makes it easy to reuse the same camera across multiple calls:

```julia
cam = (;
    camera_eye    = Vec3f(2, -3, 1.5),
    camera_lookat = Vec3f(0, 0, 0.9),
    camera_fov    = 35f0,
)

viz_motion(model, seq; cam...)
record_motion(model, seq, "out.mp4"; cam...)
render_frame(model, seq, 1, "frame.png"; cam...)
```

## Coordinate System (`seq.up`)

All formats loaded by `load_motion` (smplcodec v1/v2, AMASS npz) store motion in **Z-up** world coordinates: the root joint global orientation rotates the SMPL Y-up T-pose to Z-up, so `smpl_lbs` output vertices have Z as height. Makie's LScene camera is also Z-up by default, so motions render correctly with no manual rotation.

`seq.up` records this for the visualizer:
- `:z` (default) — Z-up; Makie renders correctly as-is
- `:y` — Y-up; visualizer rotates data to Z-up before rendering

To override for data from Y-up sources:

```julia
seq = load_motion("walk.npz")
seq_yup = MotionSequence(seq.poses, seq.betas, seq.trans, seq.fps, seq.model_type, seq.gender, :y)
```

## Scene Bounds and Ground Grid

Scene limits are computed automatically from the motion trajectory (`seq.trans`) with a margin large enough to show the full body. A ground-plane grid is drawn at foot level in all visualization functions — no manual `Rect3f` tuning needed.

## Tips

### Performance

For long sequences (> 500 frames) the interactive player recomputes LBS on every slider tick, which can be slow. Pre-bake for smoother scrubbing:

```julia
# bake_motion returns (N_v, 3, N_frames) — then pass precomputed verts to mesh!
verts = bake_motion(model, seq)
```

### WGLMakie in Notebooks

```julia
using SMPL, WGLMakie
WGLMakie.activate!()

model = create_smpl_female()
seq   = load_motion("walk.npz")
viz_motion(model, seq)   # renders inline in the notebook cell
```
```
