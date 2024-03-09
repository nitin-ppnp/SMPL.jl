using NPZ; npz=NPZ
using LinearAlgebra;
using DataDeps;

struct SMPLXdata
    v_template::Array{Float32,2}
    shapedirs::Array{Float32,2}
    posedirs::Array{Float32,2}
    J_regressor::Array{Float32,2}
    parents::Array{UInt32,1}
    lbs_weights::Array{Float32,2}
    f::Array{UInt32,2}
end


create_smplx_female() = create_smplx(joinpath(datadep"SMPLX_models","SMPLX_FEMALE.npz"));
create_smplx_male() = create_smplx(joinpath(datadep"SMPLX_models","SMPLX_MALE.npz"));
create_smplx_neutral() = create_smplx(joinpath(datadep"SMPLX_models","SMPLX_NEUTRAL.npz"));

function create_smplx(model_path)
    """
    """
    model = NPZ.npzread(model_path);
    model["kintree_table"][1,1] = 0
    model["shapedirs"] = reshape(model["shapedirs"],10475*3,:)
    model["posedirs"] = reshape(model["posedirs"],10475*3,:)'
    smplx = SMPLXdata(Float32.(model["v_template"]),
                Float32.(model["shapedirs"]),
                Float32.(model["posedirs"]),
                Float32.(model["J_regressor"]),
                UInt32.(model["kintree_table"][1,:]),
                Float32.(model["weights"]),
                model["f"].+1)        # python to julia indexing
    return smplx
end


function smpl_lbs(smplx::SMPLXdata,betas::Array{Float32,1},pose::Array{Float32,1},trans::Array{Float32,1}=zeros(Float32,3))
    """pose input (3x3)x24 : batch of 24 of 3x3 rotation matrices  """
    
    v_shaped = smplx.v_template + reshape((@view smplx.shapedirs[:,1:length(betas)]) * betas,(10475,3))
    
    J = smplx.J_regressor*v_shaped
    
    pose_view = reshape(pose,(3,55))
    rot_mats = zeros(Float32,3,3,55)
    @inbounds for i in axes(rot_mats,3)
        @views rot_mats[:,:,i] = rodrigues(pose_view[:,i])
    end
    
    pose_feature = permutedims(rot_mats[:,:,2:end],[2,1,3]) .- Matrix{Float32}(1I,3,3)
    
    pose_offsets = reshape(reshape(pose_feature,(1,:))*smplx.posedirs,(:,3))
    
    v_posed = pose_offsets + v_shaped
    
    
    J_transformed, A = rigid_transform_smplx(rot_mats,J',smplx.parents.+1)
    
    T = reshape(reshape(A,(:,55))*smplx.lbs_weights',(4,4,:))
    
    v_posed_homo = vcat(v_posed',ones(Float32,1,10475))

    v_homo = zeros(Float32,4,10475)
    @inbounds for i = 1:10475
        @views mul!(v_homo[:,i], T[:,:,i], v_posed_homo[:,i])
    end

    # v_homo = zeros(Float32,4,6890)

    # loop_einsum!(EinCode((('i','j', 'k'),('j','k')),('i','k')),(T,v_posed_homo),v_homo)

    verts = @views v_homo[1:3,:] .+ trans[:,[CartesianIndex()]]

    return verts, J_transformed[1:3,4,:].+trans[:,[CartesianIndex()]], J_transformed
        
end



function rigid_transform_smplx(rot_mats,joints,parents)
    
    rel_joints = copy(joints)
    rel_joints[:,2:end] -= joints[:,parents[2:end]]

    transforms_mat = zeros(Float32,4,4,55)

    for i = 1:55
        transforms_mat[1:3,1:3,i] = rot_mats[:,:,i]
        transforms_mat[1:3,4,i] = rel_joints[:,i]
        transforms_mat[4,4,i] = 1
    end

    # transforms_mat = cat([vcat(hcat(rot_mats[:,:,i],rel_joints[:,i]),[0 0 0 1]) for i = 1:size(rot_mats)[3]]...,dims=3)
    
    transforms = zeros(Float32,size(transforms_mat))
    transforms[:,:,1] = transforms_mat[:,:,1]
    
    for i=2:55
        transforms[:,:,i] = transforms[:,:,parents[i]] * transforms_mat[:,:,i]
    end
    
    posed_joints = copy(transforms)

    
    joints_homo = vcat(joints,zeros(Float32,1,size(joints,2)))

    init_bone = zeros(Float32,4,55)
    for i = 1:55
        init_bone[:,i] = transforms[:,:,i]*joints_homo[:,i]
    end

    transforms[:,4,:] = transforms[:,4,:] - init_bone

    return posed_joints, transforms
    
end



"""
    vertices2joints_smplx(J_regressor,vertices)

TBW
"""
function vertices2joints_smplx(J_regressor,vertices)
    
    return reshape(J_regressor*reshape(vertices,(10475,:)),(55,3,:))
    
    end


"""
    shape_correction(smplx::SMPLXdata, betas::Array{Float32,1})

TBW
"""
function shape_correction(smplx::SMPLXdata, betas::Array{Float32,1})
    v_shaped = smplx.v_template + reshape((@view smplx.shapedirs[:,1:length(betas)]) * betas,(10475,3))

end



function pivot_fk(smplx::SMPLXdata,betas::Array{Float32,1},poses::Array{Float32,2},contacts::Array{Float32,2},trans::Array{Float32,1}=zeros(Float32,3))
    """pose input (3x3)x24 : batch of 24 of 3x3 rotation matrices  """
    
    verts = zeros(3,10475,size(poses,2));
    joints = zeros(3,55,size(poses,2));
    verts[:,:,1], _, joints[:,:,1] = smpl_lbs(smplx,betas,poses[1,:],trans);

    for idx in axes(poses[2:end,:])
        v, _, j = smpl_lbs(smplx,betas,poses[idx,:])

    end


    return verts, J_transformed
        
end