# math.jl — Core mathematical primitives for LBS.
#
# Functions defined here:
#   rodrigues           — axis-angle → 3×3 rotation matrix (zero allocation)
#   quat_feat           — axis-angle → 4D quaternion feature vector (SUPR)
#   forward_kinematics  — kinematic chain forward pass for arbitrary N_j joints
#
# All functions are generic over the element type T (Float32 / Float64).
# rodrigues returns a StaticArrays.SMatrix, which is stack-allocated and inlines
# cleanly into the FK loop — no heap pressure in the hot path.
#
# forward_kinematics intentionally operates on plain CPU Arrays; the kinematic
# chain has sequential data dependencies (each joint needs its parent's result)
# and so cannot be parallelised across joints. Only O(N_j) work happens here —
# the O(N_v) skinning is handled separately in model.jl where it runs on device.

using StaticArrays
using LinearAlgebra: mul!, I


# ---------------------------------------------------------------------------
# rodrigues — Rodrigues' rotation formula
# ---------------------------------------------------------------------------

"""
    rodrigues(r) -> SMatrix{3,3,T,9}

Convert an axis-angle vector `r ∈ ℝ³` to a 3×3 rotation matrix.

Implements the Rodrigues formula (also known as the exponential map on SO(3)):

    R = I + sin(θ)·K + (1 - cos(θ))·K²

where
    θ = ‖r‖         (rotation angle)
    K = skew(r/θ)   (skew-symmetric matrix of the unit rotation axis)

The formula is equivalent to:
    R[i,j] = cos(θ)·δᵢⱼ + (1-cos(θ))·nᵢnⱼ + sin(θ)·εᵢⱼₖnₖ

where n = r/θ is the unit axis and ε is the Levi-Civita symbol.

Returns an `SMatrix{3,3,T,9}` (stack-allocated 3×3 matrix from StaticArrays).
This is zero-heap-allocation and inlines directly into the FK loop.

Numerical stability: a small ε² is added under the sqrt so the formula is
well-defined when ‖r‖ → 0 (zero rotation → identity matrix). The perturbation
is O(ε²) and does not affect direction, only magnitude near zero.

# Arguments
- `r::AbstractVector{T}`: axis-angle vector, length 3

# Returns
- `SMatrix{3,3,T,9}`: rotation matrix R ∈ SO(3)

# Example
```julia
R = rodrigues(Float32[0.1, 0.2, 0.3])   # returns SMatrix{3,3,Float32,9}
```
"""
@inline function rodrigues(r::AbstractVector{T}) where {T<:AbstractFloat}
    ε  = T(1e-8)
    # θ = ‖r‖, with ε² added under sqrt for numerical safety near zero
    θ  = sqrt(r[1]^2 + r[2]^2 + r[3]^2 + ε^2)

    # Unit rotation axis n = r / θ
    nx, ny, nz = r[1]/θ, r[2]/θ, r[3]/θ

    s  = sin(θ)                   # sin(θ)
    c  = cos(θ)                   # cos(θ)
    c1 = one(T) - c               # 1 - cos(θ)

    # Rodrigues formula expanded into 9 scalar expressions.
    # Column-major layout for SMatrix (columns are listed top-to-bottom):
    #   col 1:  R[1,1], R[2,1], R[3,1]
    #   col 2:  R[1,2], R[2,2], R[3,2]
    #   col 3:  R[1,3], R[2,3], R[3,3]
    #
    # R[i,j] = cos(θ)·δᵢⱼ + (1-cos(θ))·nᵢnⱼ + sin(θ)·εᵢⱼₖnₖ
    SMatrix{3,3,T,9}(
        c + nx*nx*c1,       ny*nx*c1 + nz*s,    nz*nx*c1 - ny*s,   # col 1 (R[:,1])
        nx*ny*c1 - nz*s,    c + ny*ny*c1,        nz*ny*c1 + nx*s,   # col 2 (R[:,2])
        nx*nz*c1 + ny*s,    ny*nz*c1 - nx*s,     c + nz*nz*c1,      # col 3 (R[:,3])
    )
end


# ---------------------------------------------------------------------------
# quat_feat — Quaternion pose feature (SUPR only)
# ---------------------------------------------------------------------------

"""
    quat_feat(r) -> SVector{4,T}

Compute the 4D quaternion feature vector used by the SUPR model's pose blend
shapes. This is NOT a standard unit quaternion — it is a modified representation
designed to be a smooth function of the axis-angle input near zero rotation.

    v_sin = sin(θ/2) · (r/‖r‖)    (3D imaginary part, scaled)
    v_cos = cos(θ/2) - 1           (real part shifted to zero for zero rotation)
    quat_feat = [v_sin; v_cos]     (4D vector)

The -1 shift makes the feature zero for zero pose (θ = 0), which is required
for the pose corrective blend shapes to have zero contribution at rest pose.

# Arguments
- `r::AbstractVector{T}`: axis-angle vector, length 3

# Returns
- `SVector{4,T}`: 4D quaternion feature
"""
@inline function quat_feat(r::AbstractVector{T}) where {T<:AbstractFloat}
    ε      = T(1e-8)
    θ      = sqrt(r[1]^2 + r[2]^2 + r[3]^2 + ε^2)
    n      = r ./ θ               # unit axis  (3,)
    half_θ = θ * T(0.5)
    v_sin  = sin(half_θ) .* n    # 3D imaginary part
    v_cos  = cos(half_θ) - one(T) # real part, shifted so quat_feat(0)=0

    # Return as SVector for zero-allocation stack usage
    SVector{4,T}(v_sin[1], v_sin[2], v_sin[3], v_cos)
end


# ---------------------------------------------------------------------------
# forward_kinematics — kinematic chain forward pass
# ---------------------------------------------------------------------------

"""
    forward_kinematics(rot_mats, J, parents) -> (G_posed, A)

Compute global joint transforms by traversing the kinematic tree defined by
`parents`. This implements the forward kinematics (FK) step of LBS:

    G_k = G_{pa(k)} · T_k^local    for k = 2, 3, ..., N_j
    G_1 = T_1^local                 (root joint)

where T_k^local = [R_k | t_k^{rel}; 0ᵀ | 1] is the local 4×4 transform
with t_k^{rel} = J_k - J_{pa(k)} (joint offset relative to parent in rest pose).

Returns two arrays:
- `G_posed`: global joint transforms G_k (used for joint positions in output)
- `A`:       LBS-ready transforms A_k = G_k · [I | -J_k; 0ᵀ | 1]
              These are the T̂_k in Eq.7 of the SMPL paper.

# Arguments
- `rot_mats::AbstractArray{T,3}`: local rotation matrices, shape (3, 3, N_j)
- `J::AbstractMatrix{T}`:         joint positions from shaped mesh, shape (3, N_j)
- `parents::Vector{Int32}`:       kinematic tree, 1-indexed; parents[1]=1 (root)

# Returns
- `G_posed ::Array{T,3}`: (4, 4, N_j) global transforms before rest-pose subtraction
- `A        ::Array{T,3}`: (4, 4, N_j) LBS-ready transforms (T̂_k in paper)

# Notes
- Always runs on CPU: the N_j-length sequential chain cannot be parallelised.
  Call `Array(rot_mats)` and `Array(J)` before this function when on GPU.
- The inner 4×4 matrix multiplications are constant-bound (1:4) and fully
  unrolled by the compiler.
"""
function forward_kinematics(
    rot_mats :: AbstractArray{T, 3},   # (3, 3, N_j) — local rotation matrices
    J        :: AbstractMatrix{T},     # (3, N_j)    — rest-pose joint positions
    parents  :: Vector{Int32},         # (N_j,)      — kinematic tree (1-indexed)
) :: Tuple{Array{T,3}, Array{T,3}} where {T}

    N_j = size(rot_mats, 3)

    # ---- Compute relative joint offsets --------------------------------
    # t_k^{rel} = J_k - J_{pa(k)}   (joint offset from parent to child)
    # Root joint (k=1): t_1^{rel} = J_1 (relative to origin)
    rel_J = copy(J)                              # (3, N_j)
    @inbounds for k in 2:N_j
        p = parents[k]
        rel_J[1, k] -= J[1, p]
        rel_J[2, k] -= J[2, p]
        rel_J[3, k] -= J[3, p]
    end

    # ---- Build local 4×4 transforms T_k^{local} = [R_k | t_k^{rel}; 0ᵀ | 1] ----
    local_T = zeros(T, 4, 4, N_j)               # (4, 4, N_j)
    @inbounds for k in 1:N_j
        # Rotation block (top-left 3×3)
        local_T[1, 1, k] = rot_mats[1, 1, k];  local_T[1, 2, k] = rot_mats[1, 2, k];  local_T[1, 3, k] = rot_mats[1, 3, k]
        local_T[2, 1, k] = rot_mats[2, 1, k];  local_T[2, 2, k] = rot_mats[2, 2, k];  local_T[2, 3, k] = rot_mats[2, 3, k]
        local_T[3, 1, k] = rot_mats[3, 1, k];  local_T[3, 2, k] = rot_mats[3, 2, k];  local_T[3, 3, k] = rot_mats[3, 3, k]
        # Translation block (top-right column)
        local_T[1, 4, k] = rel_J[1, k]
        local_T[2, 4, k] = rel_J[2, k]
        local_T[3, 4, k] = rel_J[3, k]
        # Homogeneous row
        local_T[4, 4, k] = one(T)
    end

    # ---- Accumulate global chain: G_k = G_{pa(k)} * T_k^{local} -------
    # This loop is inherently sequential: G_k depends on G_{pa(k)},
    # so it cannot be parallelised across joints.
    G = zeros(T, 4, 4, N_j)                     # (4, 4, N_j) — global transforms
    @inbounds G[:, :, 1] .= local_T[:, :, 1]    # root: G_1 = T_1^{local}
    @inbounds for k in 2:N_j
        p = parents[k]
        # 4×4 matrix multiply: G[:,:,k] = G[:,:,p] * local_T[:,:,k]
        # Bounds are constant (1:4), so the compiler fully unrolls this loop.
        for c in 1:4, r in 1:4
            acc = zero(T)
            for m in 1:4
                acc += G[r, m, p] * local_T[m, c, k]
            end
            G[r, c, k] = acc
        end
    end

    # ---- Save posed joint transforms (for output positions) ------------
    G_posed = copy(G)                            # (4, 4, N_j) — before offset removal

    # ---- Compute LBS-ready transforms A_k = G_k · [I | -J_k; 0ᵀ | 1] ----
    # Equivalent to: A_k[:,4] -= G_k[:,1:3] * J_k
    # This removes the rest-pose joint offset so vertices at rest position
    # are transformed to the origin before applying the rotation.
    @inbounds for k in 1:N_j
        # t_cancel = G_k[1:3, 1:3] * J[:, k]    (3-vector)
        t1 = G[1,1,k]*J[1,k] + G[1,2,k]*J[2,k] + G[1,3,k]*J[3,k]
        t2 = G[2,1,k]*J[1,k] + G[2,2,k]*J[2,k] + G[2,3,k]*J[3,k]
        t3 = G[3,1,k]*J[1,k] + G[3,2,k]*J[2,k] + G[3,3,k]*J[3,k]
        G[1, 4, k] -= t1
        G[2, 4, k] -= t2
        G[3, 4, k] -= t3
    end

    return G_posed, G   # G is A (LBS-ready transforms T̂_k)
end
