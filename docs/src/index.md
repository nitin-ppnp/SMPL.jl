# SMPL.jl

**SMPL.jl** is a Julia implementation of forward kinematics and linear blend skinning (LBS) for the SMPL family of parametric human body models. It supports three model variants, is GPU-agnostic, and includes a motion visualization pipeline.

## Supported Models

| Model  | Joints | Vertices | Description |
|--------|--------|----------|-------------|
| [SMPL](https://smpl.is.tue.mpg.de)   | 24 | 6,890  | Skinned Multi-Person Linear model (Loper et al., 2015) |
| [SMPLX](https://smpl-x.is.tue.mpg.de) | 55 | 10,475 | SMPL with face and hand articulation (Pavlakos et al., 2019) |
| [SUPR](https://supr.is.tue.mpg.de)   | 75 | 10,475 | Sparse Unified Part-based Representation (Osman et al., 2022) |

Each model has **female**, **male**, and **neutral** gender variants.

## Features

- **Type-stable, GPU-agnostic** — parameterized structs with `AbstractArray` dispatch; move any model to GPU with `Adapt.adapt(CuArray, model)`.
- **Paper-equation code** — the 8-step LBS pipeline follows Loper et al. notation exactly.
- **Motion playback** — load AMASS or smplcodec `.smpl` files, visualize interactively or render headlessly.
- **Zero-allocation math** — `rodrigues` returns `SMatrix{3,3}` via StaticArrays.jl.
- **Static compilation** — optional JuliaC path using a custom `.smplbin` binary format.

## Quick Start

```julia
using SMPL

# Load a model (downloads on first call — credentials required)
model = create_smplx_neutral()

# Define pose and shape
betas = zeros(Float32, 10)        # shape coefficients
theta = zeros(Float32, 165)       # axis-angle pose (55 joints × 3)
trans = zeros(Float32, 3)         # root translation

# Forward pass: returns SMPLOutput with .vertices, .joints, etc.
out = smpl_lbs(model, betas, theta, trans)

println("Vertices: ", size(out.vertices))   # (10475, 3)
println("Joints:   ", size(out.joints))     # (55, 3)
```

## Next Steps

- [Installation](@ref) — install the package and download model files
- [Tutorial](@ref) — step-by-step guide with motion loading and visualization
- [API Reference](@ref) — complete function and type reference
```
