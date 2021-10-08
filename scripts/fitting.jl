# Imports
using SMPL;
using Flux;
using CUDA;
using Flux: @epochs;
using Flux.Optimise: update!
using WGLMakie;
WGLMakie.activate!()
# using GLMakie;

# setup home directory
# home_dir = ENV["SENSEI_USERSPACE_SELF"]
home_dir = "/Users/natinsaini/"

# set device
device = gpu;

# create SMPL model
smpl  = createSMPL(joinpath(home_dir,"projects/SMPL.jl/models/basicmodel_m_lbs_10_207_0_v1.0.0.npz")) |> device;

# target pose and translation
target_pose = zeros(Float32,72) |> device;
target_trans = 5 .* ones(Float32,3) |> device;

betas = zeros(Float32,10) |> device;

# SMPL fwd pass
using BenchmarkTools
@btime smpl_lbs(smpl,betas,target_pose,target_trans);

target_V, target_J = smpl_lbs(smpl,betas,target_pose,target_trans);

target_V = target_V |> cpu;
target_J = target_J |> cpu;

# Joints and mesh visualization
meshscatter(target_J[1,:],target_J[2,:],target_J[3,:],markersize=0.05,color=target_J[3,:])
mesh!(target_V',smpl.f)


# Fitting
# initial pose and shape
init_pose = zeros(Float32,72) |> device;
init_trans = zeros(Float32,3) |> device;
# SMPL fwd pass
init_V, init_J = smpl_lbs(smpl,zeros(Float32,10),init_pose,init_trans);

# create optimizer
opt = ADAM(0.1);

# Parameters to be optimized
θ = params(init_trans);

# define loss function
function kp_loss(init_pose,init_trans)
    _,j3d = smpl_lbs(smpl,zeros(Float32,10),init_pose,init_trans);
    loss = sum((j3d-target_J).^2);
    return loss;
end

# pose and translation 
poses = copy(init_pose)
trans = copy(init_trans)

# Training loop
@epochs 100 begin
    local training_loss
    grads = gradient(θ) do 
        training_loss = kp_loss(init_pose,init_trans);
        return training_loss
    end

    println("training_loss:"*string(training_loss))

    update!(opt,θ, grads)

    poses = hcat(poses,init_pose)
    trans = hcat(trans,init_trans)

end

# 3D joints viz
scene = meshscatter(target_J[1,:],target_J[2,:],target_J[3,:],markersize=0.05,color=target_J[3,:],limits=FRect3D((0,0,0),(10,10,10)));

# get training vertices
verts = zeros(Float32,3,6890,101);
for i = 1:101
    verts[:,:,i] = smpl_lbs(smpl,zeros(Float32,10),poses[:,i],trans[:,i])[1];
end

# Make vertices observables
vert = Node(verts[:,:,1]');

# Visualize the initial mesh
mesh!(vert,smpl.f;color=:turquoise)

# Visualize the training 
WGLMakie.record(scene,"test.gif") do io
    for i=1:101
        vert[] = verts[:,:,i]'
        WGLMakie.recordframe!(io)
    end
end
