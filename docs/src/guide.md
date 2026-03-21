# Tutorial

This guide walks through the most common workflows: loading a model, running forward kinematics, loading a motion sequence, and visualizing it.

## Loading a Model

```julia
using SMPL

# SMPL (24 joints, 6890 vertices)
smpl = create_smpl_female()
smpl = create_smpl_male()
smpl = create_smpl_neutral()

# SMPLX (55 joints, 10475 vertices — adds hand + face articulation)
smplx = create_smplx_neutral()

# SUPR (75 joints, 10475 vertices — quaternion pose basis)
supr = create_supr_neutral()
```

Each function returns a `BodyModel{Float32, Matrix{Float32}}` (or `SUPRModel` for SUPR) with all arrays pre-processed and ready for `smpl_lbs`. Files are downloaded and cached on first call.

You can also load from a file path directly:

```julia
smpl = create_smpl("/path/to/SMPL_FEMALE.npz")
```

## Running the Forward Pass

`smpl_lbs` implements the 8-step linear blend skinning pipeline from Loper et al. (2015):

```julia
betas = zeros(Float32, 10)     # shape blend shape coefficients
theta = zeros(Float32, 72)     # axis-angle pose: 24 joints × 3 (SMPL)
trans = zeros(Float32, 3)      # root translation

out = smpl_lbs(smpl, betas, theta, trans)
```

The return value is a `SMPLOutput` struct with the following fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `out.vertices` | `(N_v, 3)` | Final posed + skinned vertex positions |
| `out.joints` | `(N_j, 3)` | Global joint positions |
| `out.v_shaped` | `(N_v, 3)` | Vertices after shape blend shapes only |
| `out.v_posed` | `(N_v, 3)` | Vertices after shape + pose blend shapes |
| `out.J_transforms` | `(4, 4, N_j)` | Full 4×4 joint transforms (CPU array) |
| `out.faces` | `(N_f, 3)` | Triangle face indices (reference to model) |

### Pose Dimension Reference

| Model | `length(theta)` | Joints |
|-------|----------------|--------|
| SMPL  | 72  | 24 |
| SMPLX | 165 | 55 |
| SUPR  | 228 | 76 |

### Applying Custom Shape and Pose

```julia
using Random
rng = MersenneTwister(42)

# Shape: positive betas = larger/taller body
betas = Float32.(randn(rng, 10) * 0.5)

# Pose: first 3 values = global orientation (axis-angle)
# joints 2..N_j: local joint rotations
theta = zeros(Float32, 165)
theta[4:6] .= Float32.([0.0, 0.0, 0.3])   # slight left hip rotation

out = smpl_lbs(smplx, betas, theta)   # trans defaults to zeros
```

## Loading a Motion Sequence

`load_motion` supports two formats, detected automatically:

```julia
# smplcodec .smpl format (Meshcapade)
seq = load_motion("walk.smpl")

# AMASS .npz format
seq = load_motion("/path/to/AMASS/ACCAD/Female1General/A1_-_stand_stageii.npz")
```

The returned `MotionSequence` contains:

```julia
seq.poses       # (N_frames, pose_dim)  — axis-angle per frame
seq.betas       # (N_b,)                — shape parameters
seq.trans       # (N_frames, 3)         — root translation per frame
seq.fps         # Float32               — frame rate
seq.model_type  # :smpl | :smplx | :supr
seq.gender      # :male | :female | :neutral
```

## Interactive Visualization

Load a Makie backend first, then call `viz_motion`:

```julia
using SMPL, GLMakie

model = create_smplx_neutral()
seq   = load_motion("walk.smpl")

# Opens an interactive window with a timeline slider and play/pause button
viz_motion(model, seq)
```

The player controls:
- **▶ Play / ⏸ Pause** button — starts/stops playback
- **Slider** — scrub to any frame
- **Speed menu** — 0.25×, 0.5×, 1×, 2×, 4×
- **show_skeleton** keyword — overlay joint spheres

```julia
viz_motion(model, seq; show_skeleton = true)
```

## Comparing Multiple Motions

```julia
seq1 = load_motion("walk.smpl")
seq2 = load_motion("run.smpl")

# Side-by-side panels with shared timeline
viz_motions(model, [seq1, seq2];
            layout = :sidebyside,
            labels = ["Walk", "Run"])

# All motions in one panel with distinct colors
viz_motions(model, [seq1, seq2]; layout = :overlay)
```

## Headless Rendering

Use CairoMakie for servers or CI pipelines (no display required):

```julia
using SMPL, CairoMakie

model = create_smplx_neutral()
seq   = load_motion("walk.smpl")

# Export full motion as video
record_motion(model, seq, "walk.mp4"; fps = 30, resolution = (1920, 1080))

# Export single frame as PNG
render_frame(model, seq, 1, "frame_001.png"; resolution = (1280, 720))
```

## Pre-baking Vertices

For large motions where you need random-access scrubbing, pre-compute all vertices:

```julia
# Returns Array{Float32,3} of shape (N_v, 3, N_frames)
verts = bake_motion(model, seq)

# Fast random access — no LBS recomputation
verts[:, :, 42]   # frame 42
```

Note: for SMPLX with N_frames = 1000 frames, this is ~10475 × 3 × 4 × 1000 ≈ 120 MB.
```
