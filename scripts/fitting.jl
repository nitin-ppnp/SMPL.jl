using SMPL;
using Flux;
using CUDA;
using WGLMakie;
WGLMakie.activate!()
# using GLMakie;


# home_dir = ENV["SENSEI_USERSPACE_SELF"]
home_dir = "/Users/natinsaini/"

smpl  = createSMPL(joinpath(home_dir,"projects/SMPL.jl/models/basicmodel_m_lbs_10_207_0_v1.0.0.npz")) |> SMPL.cpu;

device = Flux.cpu;

target_pose = zeros(Float32,72) |> device;
target_trans = 10 .* ones(Float32,3) |> device;

target_V, target_J = smpl_lbs(smpl,zeros(Float32,10),target_pose,target_trans);

# meshscatter(target_J[1,:],target_J[2,:],target_J[3,:],markersize=0.05,color=target_J[3,:])

# mesh(target_V',smpl.f)


#
init_pose = zeros(Float32,72) |> device;
init_trans = zeros(Float32,3) |> device;

init_V, init_J = smpl_lbs(smpl,zeros(Float32,10),init_pose,init_trans);

opt = ADAM(0.1);


using Flux.Optimise: update!

θ = params(init_trans);

function kp_loss(init_pose,init_trans)
    _,j3d = smpl_lbs(smpl,zeros(Float32,10),init_pose,init_trans);
    loss = sum((j3d-target_J).^2);
    return loss;
end

using Flux: @epochs;

poses = copy(init_pose)
trans = copy(init_trans)

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

# WGLMakie.inline!(false)
scene = meshscatter(target_J[1,:],target_J[2,:],target_J[3,:],markersize=0.05,color=target_J[3,:],limits=FRect3D((0,0,0),(10,10,10)));

verts = zeros(Float32,3,6890,101);
for i = 1:101
    verts[:,:,i] = smpl_lbs(smpl,zeros(Float32,10),poses[:,i],trans[:,i])[1];
end
vert = Node(verts[:,:,1]');
mesh!(vert,smpl.f;color=:turquoise)

WGLMakie.record(scene,"test.gif") do io
    for i=1:101
        vert[] = verts[:,:,i]'
        WGLMakie.recordframe!(io)
    end
end
