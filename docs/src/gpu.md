# GPU Support

SMPL.jl supports GPU computation via [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl). The same `smpl_lbs` function runs unchanged on any GPU backend.

## Requirements

Install your GPU backend alongside Adapt.jl:

```julia
# NVIDIA (CUDA)
] add CUDA Adapt

# AMD (ROCm)
] add AMDGPU Adapt

# Apple Metal
] add Metal Adapt
```

The `AdaptExt` package extension loads automatically when Adapt.jl is in your environment.

## Moving a Model to GPU

```julia
using SMPL, CUDA, Adapt

# Load model on CPU
model_cpu = create_smplx_neutral()

# Transfer all array fields to GPU
model_gpu = Adapt.adapt(CuArray, model_cpu)
```

`parents` (kinematic chain) and `faces` (triangle indices) always remain on CPU — they are excluded from the Adapt transfer because forward kinematics is sequential and the renderer consumes face indices on CPU.

## Running the Forward Pass on GPU

Pass GPU arrays for `β` and `θ`:

```julia
betas_cu = CuArray(zeros(Float32, 10))
theta_cu = CuArray(zeros(Float32, 165))
trans_cu = CuArray(zeros(Float32, 3))

out = smpl_lbs(model_gpu, betas_cu, theta_cu, trans_cu)
```

The output `SMPLOutput` fields (`vertices`, `joints`, `v_shaped`, `v_posed`) are `CuArray`s. `J_transforms` is always a CPU `Array{Float32,3}` because forward kinematics runs sequentially.

## GPU/CPU Boundary in `smpl_lbs`

The LBS pipeline has one deliberate CPU step: **forward kinematics**. The kinematic chain is a sequential loop over joints (each joint depends on its parent's transform), which cannot be parallelised across GPU threads. The code materialises the arrays to CPU for this step and moves the result back:

```julia
# Step 6: FK runs on CPU — sequential dependency chain
rot_cpu = Array(rot_mats)   # GPU → CPU
J_cpu   = Array(J)'
G_posed, A = forward_kinematics(rot_cpu, J_cpu, model.parents)

# Step 7: blend matrix assembled on device
A_dev = A2(reshape(A, 16, N_j))   # CPU → GPU
```

All other steps (shape blend shapes, pose blend shapes, LBS skinning) run on the device.

## Skinning Dispatch

SMPL.jl uses dual dispatch for the skinning step to select the best implementation:

```julia
# CPU path: per-vertex BLAS mul! — cache-efficient, uses OpenBLAS/MKL
_lbs_skin(T::Array{F,3}, v_h::Matrix{F})

# GPU path: fused broadcast — no scalar indexing, works on any AbstractArray
_lbs_skin(T::AbstractArray{F,3}, v_h::AbstractMatrix{F})
```

Julia's method dispatch selects the CPU path for `Array` and the GPU broadcast path for any other `AbstractArray` (CuArray, ROCArray, MtlArray, etc.).

## Performance Notes

- The GPU path is most beneficial for large batches. For single frames, CPU is often faster due to the FK CPU round-trip overhead.
- For motion rendering (many frames), consider `bake_motion` on CPU with BLAS, which is well-optimised and avoids GPU transfer overhead.
- `rodrigues` uses `StaticArrays.SMatrix{3,3}` — zero allocation, fully inlined, equally efficient on CPU and GPU.
```
