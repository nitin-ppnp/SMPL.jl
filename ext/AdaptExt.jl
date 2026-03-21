# ext/AdaptExt.jl — Adapt.jl integration for BodyModel and SUPRModel.
#
# This is a Julia package extension (Julia 1.9+). It is loaded automatically
# when both SMPL.jl and Adapt.jl are present in the environment. It is NOT
# loaded in the JuliaC static compilation project (static_project/Project.toml
# does not list Adapt), so the trimmer never encounters Adapt's runtime code.
#
# Usage (after loading CUDA.jl or another GPU backend):
#
#   using CUDA, Adapt
#   cpu_model = create_smpl_neutral()
#   gpu_model = Adapt.adapt(CuArray, cpu_model)   # moves Float32 matrices to GPU
#   out       = smpl_lbs(gpu_model, β_cu, θ_cu)   # runs entirely on GPU
#
# The manual adapt_structure override keeps `parents` and `faces` on CPU because:
#   - parents: only consumed by forward_kinematics, which runs sequentially on CPU
#   - faces:   only consumed by the renderer / visualization, not by LBS

module AdaptExt

using SMPL
using Adapt

# BodyModel: adapt all AbstractMatrix fields to the target array type.
# parents (Vector{Int32}) and faces (Matrix{UInt32}) are always kept on CPU.
function Adapt.adapt_structure(to, m::SMPL.BodyModel)
    SMPL.BodyModel(
        Adapt.adapt(to, m.v_template),    # move to GPU
        Adapt.adapt(to, m.shapedirs),     # move to GPU
        Adapt.adapt(to, m.posedirs),      # move to GPU
        Adapt.adapt(to, m.J_regressor),   # move to GPU
        Adapt.adapt(to, m.lbs_weights),   # move to GPU
        m.parents,                        # stay on CPU (sequential FK)
        m.faces,                          # stay on CPU (renderer use)
    )
end

# SUPRModel: same pattern; J_bias is also an AbstractMatrix and moves to GPU.
function Adapt.adapt_structure(to, m::SMPL.SUPRModel)
    SMPL.SUPRModel(
        Adapt.adapt(to, m.v_template),
        Adapt.adapt(to, m.shapedirs),
        Adapt.adapt(to, m.posedirs),
        Adapt.adapt(to, m.J_regressor),
        Adapt.adapt(to, m.J_bias),
        Adapt.adapt(to, m.lbs_weights),
        m.parents,                        # stay on CPU
        m.faces,                          # stay on CPU
    )
end

end  # module AdaptExt
