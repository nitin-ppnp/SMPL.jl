# API Reference

## Data Structures

```@docs
SMPL.BodyModel
SMPL.SUPRModel
SMPL.SMPLOutput
SMPL.MotionSequence
```

## Model Constructors

### SMPL

```@docs
SMPL.create_smpl
SMPL.create_smpl_female
SMPL.create_smpl_male
SMPL.create_smpl_neutral
```

### SMPLX

```@docs
SMPL.create_smplx
SMPL.create_smplx_female
SMPL.create_smplx_male
SMPL.create_smplx_neutral
```

### SUPR

```@docs
SMPL.create_supr
SMPL.create_supr_female
SMPL.create_supr_male
SMPL.create_supr_neutral
```

## Forward Pass

```@docs
SMPL.smpl_lbs
SMPL.pivot_fk
```

## Math Primitives

```@docs
SMPL.rodrigues
SMPL.quat_feat
SMPL.forward_kinematics
```

## Motion IO

```@docs
SMPL.load_motion
SMPL.load_pivot_labels
```

## Visualization

!!! note
    The following functions are available only when a Makie backend
    (`GLMakie`, `CairoMakie`, or `WGLMakie`) is loaded.

```@docs
SMPL.bake_motion
SMPL.viz_motion
SMPL.viz_motions
SMPL.record_motion
SMPL.render_frame
```
```
