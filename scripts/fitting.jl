using SMPL;
using Flux;
using CUDA;
using WGLMakie;
WGLMakie.activate!()


home_dir = ENV["SENSEI_USERSPACE_SELF"]

smpl  = createSMPL(joinpath(home_dir,"projects/SMPL.jl/models/basicmodel_m_lbs_10_207_0_v1.0.0.npz")) |> SMPL.cpu;

device = Flux.cpu;

target_pose = zeros(Float32,72) |> device;
target_trans = ones(Float32,3) |> device;

target_V, target_J = smpl_lbs(smpl,zeros(Float32,10),target_pose,target_trans);

# meshscatter(target_J[1,:],target_J[2,:],target_J[3,:],markersize=0.05,color=target_J[3,:])

# mesh(target_V',smpl.f)


#
init_pose = zeros(Float32,72) |> device;
init_trans = zeros(Float32,3) |> device;

init_V, init_J = smpl_lbs(smpl,zeros(Float32,10),init_pose,init_trans);

opt = ADAM(0.001);


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
# scene = meshscatter(target_J[1,:],target_J[2,:],target_J[3,:],markersize=0.05,color=target_J[3,:]);
verts,_ = smpl_lbs(smpl,zeros(Float32,10,101),poses,trans);
vert = Node(verts[:,:,1]');
scene = mesh(vert,smpl.f;color=:turquoise);

WGLMakie.record(scene,"test.gif",1:size(verts,3)) do i
    vert[] = verts[:,:,i]'
end
