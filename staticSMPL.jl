using LinearAlgebra;

include("src/static_smplfuncs.jl")
include("src/utils.jl")

const DEFAULT_MODEL_PATH = "C:\\Users\\nitin\\Desktop\\projects\\SMPL.jl\\SMPL_MALE.smplbin"

# Bypass Julia's IO dispatch (trimmed in unsafe mode) — call C puts directly.
# puts(s) writes s to stdout and appends a newline.
function p(s::String)
    ccall(:puts, Cint, (Cstring,), s)
end

function @main(args::Vector{String})::Cint
    model_path = length(args) >= 1 ? args[1] : DEFAULT_MODEL_PATH

    if !isfile(model_path)
        p(string("Error: model file not found: ", model_path))
        p("Usage: smpl.exe [path/to/SMPL_MALE.smplbin]")
        p("Convert from .npz first:  julia scripts/convert_model.jl")
        return 1
    end

    p(string("Loading model from: ", model_path))
    smpl = create_smpl(model_path)

    betas = MallocVector{Float32}(undef, 10); fill!(betas, 0f0)
    pose  = MallocVector{Float32}(undef, 72); fill!(pose,  0f0)
    trans = MallocVector{Float32}(undef,  3); fill!(trans, 0f0)

    verts, joints = smpl_lbs(smpl, betas, pose, trans)

    p("=== SMPL Forward Pass ===")
    p(string("Vertices : ", size(verts, 2), " (3 x ", size(verts, 2), ")"))
    p(string("Joints   : ", size(joints, 2), " (3 x ", size(joints, 2), ")"))

    p("\n--- First 5 vertices (x, y, z) ---")
    for i in 1:5
        p(string("  v[", i, "]: (",
                 round(verts[1,i]; digits=4), ", ",
                 round(verts[2,i]; digits=4), ", ",
                 round(verts[3,i]; digits=4), ")"))
    end

    p("\n--- All 24 joint positions (x, y, z) ---")
    joint_names = ["Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee",
                   "Spine2","L_Ankle","R_Ankle","Spine3","L_Foot","R_Foot",
                   "Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder",
                   "L_Elbow","R_Elbow","L_Wrist","R_Wrist","L_Hand","R_Hand"]
    for i in 1:24
        p(string("  ", rpad(joint_names[i], 12), ": (",
                 round(joints[1,i]; digits=4), ", ",
                 round(joints[2,i]; digits=4), ", ",
                 round(joints[3,i]; digits=4), ")"))
    end

    vx = @view verts[1,:]
    vy = @view verts[2,:]
    vz = @view verts[3,:]
    p("\n--- Vertex bounding box ---")
    p(string("  X: [", round(minimum(vx); digits=4), ", ", round(maximum(vx); digits=4), "]"))
    p(string("  Y: [", round(minimum(vy); digits=4), ", ", round(maximum(vy); digits=4), "]"))
    p(string("  Z: [", round(minimum(vz); digits=4), ", ", round(maximum(vz); digits=4), "]"))

    return 0
end
