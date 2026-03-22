# model.jl — LBS forward pass for SMPL/SMPLX (BodyModel) and SUPR (SUPRModel).
#
# Public API:
#   smpl_lbs(model::BodyModel, β, θ, trans)    -> SMPLOutput
#   smpl_lbs(model::SUPRModel, β, θ, trans)    -> SMPLOutput
#   pivot_fk(model, β, poses, contacts, trans) -> (vertices, joints)
#
# The two `smpl_lbs` methods are resolved statically at compile time based on
# the first argument type — no dynamic dispatch, no Dict, fully type-stable.
#
# GPU compatibility:
#   All O(N_v) operations use AbstractArray broadcasting, which runs on any
#   backend (CUDA, Metal, ROCm). The one exception is forward_kinematics, which
#   is sequential and always runs on CPU; the O(N_j) arrays it needs are
#   materialized with Array() before the call and moved back with A2() after.
#
# Conventions:
#   Notation follows the SMPL paper (Loper et al., 2015):
#     β = shape coefficients (betas)
#     θ = pose coefficients in axis-angle (pose)
#     ψ = pose feature vector
#     v̄ = mean shape template
#     S, P = shape / pose blend shape bases
#     W = LBS weights
#     G_k = global joint transform for joint k
#     A_k = LBS-ready transform (G_k with rest-pose offset removed)

using LinearAlgebra: mul!, I


# ---------------------------------------------------------------------------
# _lbs_skin — per-vertex 4×4 transform application
# ---------------------------------------------------------------------------

# CPU path: column-wise BLAS mul! avoids allocating the (4,4,N_v) intermediate.
# This method is more specific (Array) so it wins over the AbstractArray fallback.
function _lbs_skin(T_blend::Array{F,3}, v_h::Matrix{F}) :: Matrix{F} where {F}
    out = similar(v_h)                   # (4, N_v)
    @inbounds for i in axes(out, 2)
        # In-place multiply: out[:,i] = T_blend[:,:,i] * v_h[:,i]
        mul!(@view(out[:, i]), @view(T_blend[:, :, i]), @view(v_h[:, i]))
    end
    return out
end

# GPU / generic path: fused broadcast.
# T_blend is (4, 4, N_v) and v_h is (4, N_v).
# The reshape broadcasts v_h across the first dim, then sums over the matrix dim.
# On CUDA/Metal this becomes a single fused kernel with no extra allocations
# beyond the output.
function _lbs_skin(T_blend::AbstractArray{F,3}, v_h::AbstractMatrix{F}) :: AbstractMatrix{F} where {F}
    # result[k, i] = Σ_j  T_blend[k, j, i] * v_h[j, i]
    dropdims(
        sum(T_blend .* reshape(v_h, 1, size(v_h, 1), size(v_h, 2)), dims=2),
        dims=2,
    )
end


# ---------------------------------------------------------------------------
# smpl_lbs for BodyModel — SMPL and SMPLX
# ---------------------------------------------------------------------------

"""
    smpl_lbs(model::BodyModel{T,A2}, β, θ, trans) -> SMPLOutput

Linear Blend Skinning forward pass for SMPL (24 joints) or SMPLX (55 joints).

Implements equations 1–8 from Loper et al., "SMPL: A Skinned Multi-Person
Linear Model", SIGGRAPH Asia 2015. Variable names match the paper directly.

# Arguments
- `model`: loaded body model (CPU or GPU, both work identically)
- `β`:     shape coefficients, length ≤ size(model.shapedirs, 2)  (N_b,)
- `θ`:     axis-angle pose, length = N_j * 3                       (N_j*3,)
- `trans`: global translation, length 3  [default: zeros]

# Returns
`SMPLOutput` with fields: `vertices` (N_v,3), `joints` (N_j,3),
`v_shaped` (N_v,3), `v_posed` (N_v,3), `J_transforms` (4,4,N_j), `faces`.

# GPU usage
```julia
gpu_model = Adapt.adapt(CuArray, model)   # move to GPU (requires CUDA.jl)
β_gpu     = CuArray(β)
θ_gpu     = CuArray(θ)
out       = smpl_lbs(gpu_model, β_gpu, θ_gpu)  # runs on GPU
```
"""
function smpl_lbs(
    model :: BodyModel{ET, A2},
    β     :: AbstractVector{ET},
    θ     :: AbstractVector{ET},
    trans :: AbstractVector{ET} = zeros(ET, 3),
) :: SMPLOutput{ET, A2} where {ET, A2 <: AbstractMatrix{ET}}

    N_v = size(model.v_template, 1)   # number of vertices (6890 or 10475)
    N_j = size(model.J_regressor, 1)  # number of joints   (24 or 55)
    N_b = length(β)                   # number of shape components used

    # ------------------------------------------------------------------
    # (1)  v_s = v̄ + S·β        — shape blend shapes
    #      Each principal component S[:,n] is a (N_v*3,) deformation field.
    #      The product S[:,1:N_b]·β adds a weighted combination to the template.
    # ------------------------------------------------------------------
    v_shaped = model.v_template .+
               reshape((@view model.shapedirs[:, 1:N_b]) * β, N_v, 3)  # (N_v, 3)

    # ------------------------------------------------------------------
    # (2)  J = R_J · v_s         — joint positions from shaped mesh
    #      J_regressor is a learned matrix that regresses joint positions
    #      from vertex positions via per-vertex weights.
    # ------------------------------------------------------------------
    J = model.J_regressor * v_shaped    # (N_j, 3)

    # ------------------------------------------------------------------
    # (3)  R_k = rodrigues(θ_k)  — local rotation matrix per joint
    #      θ is a flat axis-angle vector; reshape to (3, N_j) for easy slicing.
    #      Each column θ_mat[:,k] is the axis-angle for joint k.
    #      Array(θ) is a no-op on CPU; on GPU it brings θ to CPU so that
    #      rodrigues can access elements without triggering slow scalar indexing.
    # ------------------------------------------------------------------
    θ_cpu    = Array(θ)                          # (N_j*3,) on CPU
    θ_mat    = reshape(θ_cpu, 3, N_j)            # (3, N_j) — view, no copy
    rot_mats = zeros(ET, 3, 3, N_j)             # (3, 3, N_j) — local rotations, CPU
    @inbounds for k in axes(rot_mats, 3)
        rot_mats[:, :, k] .= rodrigues(@view θ_mat[:, k])
    end

    # ------------------------------------------------------------------
    # (4)  ψ = vec(R_{2..K}ᵀ − I)  — pose feature vector
    #      The pose corrective captures surface deformation from joint rotation.
    #      Only joints 2..N_j are used (root rotation has no shape corrective).
    #      Transposing R before subtracting I follows the SMPL paper convention.
    #      Use an explicit 3×3 identity (not UniformScaling) for 3D broadcasting.
    # ------------------------------------------------------------------
    I3 = Matrix{ET}(I, 3, 3)
    ψ  = reshape(
        permutedims(rot_mats[:, :, 2:end], (2, 1, 3)) .- I3,  # (3, 3, N_j-1)
        1, :,                                                   # (1, (N_j-1)*9)
    )

    # ------------------------------------------------------------------
    # (5)  v_p = v_s + P·ψ       — pose blend shapes
    #      P (posedirs) maps the ((N_j-1)*9,) pose feature to a (N_v*3,) offset.
    #      A2(ψ) moves ψ to the model's device (no-op on CPU); required so that
    #      ψ * model.posedirs does not mix CPU and GPU arrays.
    # ------------------------------------------------------------------
    v_posed = v_shaped .+ reshape(A2(ψ) * model.posedirs, N_v, 3)  # (N_v, 3)

    # ------------------------------------------------------------------
    # (6)  G_k = FK(R, J, pa)    — forward kinematics
    #      The kinematic chain is sequential: G_k = G_{pa(k)} * T_k^{local}.
    #      rot_mats is already on CPU (populated by the rodrigues loop above).
    # ------------------------------------------------------------------
    rot_cpu = rot_mats                 # (3, 3, N_j) on CPU — already CPU
    J_cpu   = Array(J)'                # (3, N_j)    on CPU, transposed for FK
    G_posed, A = forward_kinematics(rot_cpu, J_cpu, model.parents)

    # ------------------------------------------------------------------
    # (7)  T_i = Σ_k w_{ki}·G_k   — per-vertex blend matrix
    #      Reshape A to (16, N_j), multiply by LBS weights (N_j, N_v),
    #      reshape result to (4, 4, N_v): one 4×4 matrix per vertex.
    #      A2(...) moves A back to the model's device (no-op on CPU).
    # ------------------------------------------------------------------
    A_dev   = A2(reshape(A, 16, N_j))                          # (16, N_j) on device
    T_blend = reshape(A_dev * model.lbs_weights', 4, 4, N_v)   # (4, 4, N_v)

    # ------------------------------------------------------------------
    # (8)  v_i = T_i · [v_p_i; 1]  — linear blend skinning
    #      Append row of ones for homogeneous coordinates (handles translation).
    #      Use similar(model.v_template, 1, N_v) to allocate the ones row on the
    #      same device as v_posed (GPU or CPU), avoiding a CPU/GPU vcat mismatch.
    #      _lbs_skin dispatches to the CPU loop or GPU broadcast automatically.
    # ------------------------------------------------------------------
    ones_row  = fill!(similar(model.v_template, 1, N_v), one(ET))  # (1, N_v) on device
    v_posed_h = vcat(v_posed', ones_row)                            # (4, N_v) homogeneous
    v_h       = _lbs_skin(T_blend, v_posed_h)                      # (4, N_v) skinned

    # Extract 3D positions and apply global translation.
    # trans may arrive as a CPU array even for GPU models (user convenience).
    # reshape trick: A2 is always 2D, so construct (3,1) then reshape to (3,).
    trans_dev = reshape(A2(reshape(collect(ET, trans), 3, 1)), 3)   # (3,) on device
    verts  = (v_h[1:3, :] .+ trans_dev)'             # (N_v, 3)
    joints = (A2(G_posed[1:3, 4, :]) .+ trans_dev)'  # (N_j, 3)

    return SMPLOutput{ET, A2}(
        verts,
        joints,
        v_shaped,
        v_posed,
        G_posed,        # (4, 4, N_j) — always a CPU Array
        model.faces,
    )
end


# ---------------------------------------------------------------------------
# smpl_lbs for SUPRModel — SUPR
# ---------------------------------------------------------------------------

"""
    smpl_lbs(model::SUPRModel{T,A2}, β, θ, trans) -> SMPLOutput

Linear Blend Skinning forward pass for SUPR (75 joints, 10475 vertices).

SUPR uses the same overall LBS pipeline as SMPL but differs in two steps:

  Step 2: Generalized affine joint regressor.
          `J = reshape(K·v_flat, 3, N_j)' + J_bias`
          where K is (3*N_j, N_v*3) and v_flat = reshape(v_shaped', :).

  Steps 3–4: Quaternion pose features instead of rotation-matrix features.
          `quat_feat(θ_k)` returns a 4D vector per joint → ψ is (N_j*4,).

All other steps (shape blend, FK, LBS skinning) are identical to SMPL.
"""
function smpl_lbs(
    model :: SUPRModel{ET, A2},
    β     :: AbstractVector{ET},
    θ     :: AbstractVector{ET},
    trans :: AbstractVector{ET} = zeros(ET, 3),
) :: SMPLOutput{ET, A2} where {ET, A2 <: AbstractMatrix{ET}}

    N_v = size(model.v_template, 1)   # 10475
    N_j = size(model.lbs_weights, 2)  # 75
    N_b = length(β)

    # ------------------------------------------------------------------
    # (1)  v_s = v̄ + S·β        — shape blend shapes (same as SMPL)
    # ------------------------------------------------------------------
    v_shaped = model.v_template .+
               reshape((@view model.shapedirs[:, 1:N_b]) * β, N_v, 3)  # (N_v, 3)

    # ------------------------------------------------------------------
    # (2)  J = K·v_flat + b      — generalized affine joint positions
    #      SUPR's joint regressor is a full affine map from all vertex
    #      coordinates (not just scalar per-vertex weights like SMPL).
    #      v_flat groups coordinates as [x_1..x_{N_v}, y_1..y_{N_v}, z_1..z_{N_v}]
    #      because it's formed from transpose(v_shaped) then flattened.
    # ------------------------------------------------------------------
    v_flat = reshape(v_shaped', :)                           # (N_v*3,)
    J      = reshape(model.J_regressor * v_flat, 3, N_j)' .+ model.J_bias  # (N_j, 3)

    # ------------------------------------------------------------------
    # (3)  R_k = rodrigues(θ_k),  ψ_k = quat_feat(θ_k)
    #      SUPR uses 4D quaternion features for pose blend shapes.
    #      Both rot_mats (for FK) and quat_feats (for pose correction)
    #      are computed from the same axis-angle input θ.
    #      Array(θ) is a no-op on CPU; on GPU it brings θ to CPU so that
    #      rodrigues / quat_feat can access elements without slow scalar indexing.
    # ------------------------------------------------------------------
    θ_cpu      = Array(θ)                      # (N_j*3,) on CPU
    θ_mat      = reshape(θ_cpu, 3, N_j)        # (3, N_j) — view, no copy
    rot_mats   = zeros(ET, 3, 3, N_j)         # (3, 3, N_j) — local rotations, CPU
    quat_feats = zeros(ET, 4, N_j)            # (4, N_j)  — quaternion features, CPU
    @inbounds for k in axes(rot_mats, 3)
        rot_mats[:, :, k] .= rodrigues(@view θ_mat[:, k])
        quat_feats[:, k]   = quat_feat(@view θ_mat[:, k])
    end

    # ------------------------------------------------------------------
    # (4)  ψ = vec(quat_feats)   — pose feature vector (1, N_j*4)
    #      SUPR uses all N_j joints (including root), unlike SMPL which
    #      skips the root rotation in the pose corrective.
    # ------------------------------------------------------------------
    ψ = reshape(quat_feats, 1, :)              # (1, N_j*4)

    # ------------------------------------------------------------------
    # (5)  v_p = v_s + P·ψ       — pose blend shapes
    #      posedirs is (N_j*4, N_v*3); result reshaped to (N_v, 3).
    #      A2(ψ) moves ψ to the model's device (no-op on CPU); required so that
    #      ψ * model.posedirs does not mix CPU and GPU arrays.
    # ------------------------------------------------------------------
    v_posed = v_shaped .+ reshape(A2(ψ) * model.posedirs, N_v, 3)   # (N_v, 3)

    # ------------------------------------------------------------------
    # (6)  FK — same sequential chain as SMPL, runs on CPU
    #      rot_mats is already on CPU (populated by the rodrigues loop above).
    # ------------------------------------------------------------------
    rot_cpu = rot_mats                         # (3, 3, N_j) on CPU — already CPU
    J_cpu   = Array(J)'                        # (3, N_j) for FK convention
    G_posed, A = forward_kinematics(rot_cpu, J_cpu, model.parents)

    # ------------------------------------------------------------------
    # (7)  T_i = Σ_k w_{ki}·G_k — per-vertex blend matrix (same as SMPL)
    # ------------------------------------------------------------------
    A_dev   = A2(reshape(A, 16, N_j))
    T_blend = reshape(A_dev * model.lbs_weights', 4, 4, N_v)     # (4, 4, N_v)

    # ------------------------------------------------------------------
    # (8)  v_i = T_i · [v_p_i; 1] — LBS skinning (same as SMPL)
    # ------------------------------------------------------------------
    ones_row  = fill!(similar(model.v_template, 1, N_v), one(ET))  # (1, N_v) on device
    v_posed_h = vcat(v_posed', ones_row)                            # (4, N_v)
    v_h       = _lbs_skin(T_blend, v_posed_h)                      # (4, N_v)

    trans_dev = reshape(A2(reshape(collect(ET, trans), 3, 1)), 3)   # (3,) on device
    verts  = (v_h[1:3, :] .+ trans_dev)'                            # (N_v, 3)
    joints = (A2(G_posed[1:3, 4, :]) .+ trans_dev)'                 # (N_j, 3)

    return SMPLOutput{ET, A2}(
        verts,
        joints,
        v_shaped,
        v_posed,
        G_posed,
        model.faces,
    )
end


# ---------------------------------------------------------------------------
# pivot_fk — contact-constrained forward kinematics over a pose sequence
# ---------------------------------------------------------------------------

"""
    pivot_fk(model, β, poses, contacts, trans) -> (vertices, joints)

Contact-constrained forward kinematics over a sequence of poses.

At each frame, the contact joint (highest contact weight) is held fixed in
world space. The remaining body is placed relative to that anchor using the
global joint transforms from `smpl_lbs`.

# Arguments
- `model`:    BodyModel or SUPRModel
- `β`:        shape coefficients                      (N_b,)
- `poses`:    axis-angle pose sequence, one per column (N_j*3, N_frames)
- `contacts`: contact weight per joint per frame      (N_j, N_frames)
- `trans`:    initial global translation              (3,)  [default: zeros]

# Returns
- `vertices`: (3, N_v, N_frames) vertex positions in world space
- `joints`:   (4, 4, N_j, N_frames) global 4×4 joint transforms
"""
function pivot_fk(
    model    :: Union{BodyModel{ET, A2}, SUPRModel{ET, A2}},
    β        :: AbstractVector{ET},
    poses    :: AbstractMatrix{ET},      # (N_j*3, N_frames)
    contacts :: AbstractMatrix{ET},      # (N_j, N_frames)
    trans    :: AbstractVector{ET} = zeros(ET, 3),
) where {ET, A2 <: AbstractMatrix{ET}}

    N_j      = size(model.lbs_weights, 2)
    N_v      = size(model.v_template, 1)
    N_frames = size(poses, 2)

    verts  = zeros(ET, 3, N_v, N_frames)       # (3, N_v, N_frames)
    joints = zeros(ET, 4, 4, N_j, N_frames)    # (4, 4, N_j, N_frames)

    # First frame: straight LBS with global translation
    ot   = smpl_lbs(model, β, poses[:, 1], trans)
    j_ot = ot.J_transforms                      # (4, 4, N_j) — global transforms

    verts[:, :, 1]      = ot.vertices'
    joints[:, :, :, 1]  = j_ot

    # Contact joint: index of highest-weight contact joint, per frame
    contact_joints = argmax(contacts, dims=1)   # (1, N_frames)

    for (idx, cj) in enumerate(contact_joints[1:end-1])
        ot_fut   = smpl_lbs(model, β, poses[:, idx+1])
        j_ot_fut = ot_fut.J_transforms             # (4, 4, N_j)
        v_ot_fut = ot_fut.vertices'                # (3, N_v)

        # Express all joints relative to the contact joint in the future frame
        cj_inv         = inv(j_ot_fut[:, :, cj[1]])
        rel_joints_fut = stack([cj_inv * j_ot_fut[:, :, i] for i in axes(j_ot_fut, 3)])

        # Relative transform: past contact joint → future contact joint
        joint_rel_past = inv(j_ot[:, :, cj[1]]) * j_ot_fut[:, :, cj[1]]

        # Zero translation: contact joint stays fixed in world space
        joint_rel_past[1:3, 4] .= 0

        # Compose: anchor future pose at past contact joint location
        rotated_fut = joints[:, :, cj[1], idx] * joint_rel_past
        joints[:, :, :, idx+1] = stack([
            rotated_fut * rel_joints_fut[:, :, i]
            for i in axes(rel_joints_fut, 3)
        ])

        # Vertices in future frame, expressed relative to the contact joint
        verts_wrt_cj = cj_inv[1:3, 1:3] * v_ot_fut .+ cj_inv[1:3, 4]
        verts[:, :, idx+1] = (
            joints[1:3, 1:3, cj[1], idx+1] * verts_wrt_cj
            .+ joints[1:3, 4, cj[1], idx+1]
        )

        j_ot = j_ot_fut
    end

    return verts, joints
end
